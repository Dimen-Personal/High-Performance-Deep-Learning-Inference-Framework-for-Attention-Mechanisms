#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


// ============================================================
// CUDA error helper
// ============================================================

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t error = (call);                                          \
        if(error != cudaSuccess)                                             \
        {                                                                    \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ") + cudaGetErrorString(error)      \
            );                                                               \
        }                                                                    \
    } while(false)


// ============================================================
// Tunable AttentionSoftmax kernel
//
// Input:
//     scores  [seq_len, seq_len]
//
// Output:
//     weights [seq_len, seq_len]
//
// Each CUDA block handles one row.
//
// Only blockDim.x changes.
//
// The algorithm is intentionally kept equivalent to the
// current project implementation.
// ============================================================

__global__
void attention_softmax_tuning_kernel(
    const float* scores,
    float* weights,
    int seq_len
)
{
    extern __shared__ float shared[];


    const int row =
        blockIdx.x;


    const int tid =
        threadIdx.x;


    if(row >= seq_len)
        return;


    const int row_offset =
        row * seq_len;


    // ========================================================
    // Step 1:
    // Find row maximum
    // ========================================================

    float local_max =
        -FLT_MAX;


    for(
        int col = tid;
        col < seq_len;
        col += blockDim.x
    )
    {
        local_max =
            fmaxf(
                local_max,
                scores[
                    row_offset
                    + col
                ]
            );
    }


    shared[tid] =
        local_max;


    __syncthreads();


    // ========================================================
    // Reduce maximum
    // ========================================================

    for(
        int stride =
            blockDim.x / 2;

        stride > 0;

        stride /= 2
    )
    {
        if(tid < stride)
        {
            shared[tid] =
                fmaxf(
                    shared[tid],
                    shared[
                        tid + stride
                    ]
                );
        }


        __syncthreads();
    }


    const float max_value =
        shared[0];


    // ========================================================
    // Step 2:
    // exp(x - max)
    //
    // Each thread calculates a partial sum.
    // ========================================================

    float local_sum =
        0.0f;


    for(
        int col = tid;
        col < seq_len;
        col += blockDim.x
    )
    {
        const float value =
            expf(
                scores[
                    row_offset
                    + col
                ]
                -
                max_value
            );


        weights[
            row_offset
            + col
        ] =
            value;


        local_sum +=
            value;
    }


    shared[tid] =
        local_sum;


    __syncthreads();


    // ========================================================
    // Reduce sum
    // ========================================================

    for(
        int stride =
            blockDim.x / 2;

        stride > 0;

        stride /= 2
    )
    {
        if(tid < stride)
        {
            shared[tid] +=
                shared[
                    tid + stride
                ];
        }


        __syncthreads();
    }


    const float total_sum =
        shared[0];


    // ========================================================
    // Step 3:
    // Normalize
    // ========================================================

    for(
        int col = tid;
        col < seq_len;
        col += blockDim.x
    )
    {
        weights[
            row_offset
            + col
        ] /=
            total_sum;
    }
}


// ============================================================
// Launch helper
// ============================================================

void launch_attention_softmax(
    const float* scores,
    float* weights,
    int seq_len,
    int block_size,
    cudaStream_t stream = nullptr
)
{
    const std::size_t shared_bytes =
        static_cast<std::size_t>(
            block_size
        )
        *
        sizeof(float);


    attention_softmax_tuning_kernel<<<
        seq_len,
        block_size,
        shared_bytes,
        stream
    >>>(
        scores,
        weights,
        seq_len
    );
}


// ============================================================
// Benchmark result
// ============================================================

struct BenchmarkResult
{
    int seq_len =
        0;

    int block_size =
        0;

    float average_ms =
        0.0f;

    float relative_to_256 =
        0.0f;

    bool correct =
        false;
};


// ============================================================
// Initialize scores
//
// We generate deterministic values.
//
// Values above diagonal are set to -INFINITY to reproduce
// causal AttentionScores input.
// ============================================================

void initialize_scores(
    std::vector<float>& scores,
    int seq_len
)
{
    for(
        int row = 0;
        row < seq_len;
        row++
    )
    {
        for(
            int col = 0;
            col < seq_len;
            col++
        )
        {
            const std::size_t index =
                static_cast<std::size_t>(
                    row
                )
                *
                seq_len
                +
                col;


            if(col > row)
            {
                scores[index] =
                    -INFINITY;
            }
            else
            {
                const int value =
                    (
                        row * 17
                        +
                        col * 13
                    )
                    % 101
                    -
                    50;


                scores[index] =
                    static_cast<float>(
                        value
                    )
                    *
                    0.02f;
            }
        }
    }
}


// ============================================================
// Compare outputs
// ============================================================

bool compare_outputs(
    const std::vector<float>& reference,
    const std::vector<float>& output,
    float tolerance
)
{
    if(
        reference.size()
        !=
        output.size()
    )
    {
        return false;
    }


    for(
        std::size_t i = 0;
        i < reference.size();
        i++
    )
    {
        const float diff =
            std::fabs(
                reference[i]
                -
                output[i]
            );


        if(diff > tolerance)
        {
            std::cerr
                << "Mismatch at index "
                << i
                << ": reference = "
                << reference[i]
                << ", output = "
                << output[i]
                << ", diff = "
                << diff
                << std::endl;


            return false;
        }
    }


    return true;
}


// ============================================================
// Benchmark one configuration
// ============================================================

float benchmark_config(
    const float* d_scores,
    float* d_weights,
    int seq_len,
    int block_size,
    int warmup_runs,
    int benchmark_runs
)
{
    // ========================================================
    // Warmup
    // ========================================================

    for(
        int i = 0;
        i < warmup_runs;
        i++
    )
    {
        launch_attention_softmax(
            d_scores,
            d_weights,
            seq_len,
            block_size
        );
    }


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // CUDA Events
    // ========================================================

    cudaEvent_t start;

    cudaEvent_t stop;


    CUDA_CHECK(
        cudaEventCreate(
            &start
        )
    );


    CUDA_CHECK(
        cudaEventCreate(
            &stop
        )
    );


    CUDA_CHECK(
        cudaEventRecord(
            start
        )
    );


    for(
        int i = 0;
        i < benchmark_runs;
        i++
    )
    {
        launch_attention_softmax(
            d_scores,
            d_weights,
            seq_len,
            block_size
        );
    }


    CUDA_CHECK(
        cudaEventRecord(
            stop
        )
    );


    CUDA_CHECK(
        cudaEventSynchronize(
            stop
        )
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    float total_ms =
        0.0f;


    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );


    CUDA_CHECK(
        cudaEventDestroy(
            start
        )
    );


    CUDA_CHECK(
        cudaEventDestroy(
            stop
        )
    );


    return
        total_ms
        /
        static_cast<float>(
            benchmark_runs
        );
}


// ============================================================
// Run one sequence length
// ============================================================

std::vector<BenchmarkResult>
run_sequence_length(
    int seq_len,
    const std::vector<int>& block_sizes,
    int warmup_runs,
    int benchmark_runs
)
{
    const std::size_t matrix_size =
        static_cast<std::size_t>(
            seq_len
        )
        *
        static_cast<std::size_t>(
            seq_len
        );


    // ========================================================
    // Host memory
    // ========================================================

    std::vector<float> h_scores(
        matrix_size
    );


    std::vector<float> h_reference(
        matrix_size
    );


    std::vector<float> h_output(
        matrix_size
    );


    initialize_scores(
        h_scores,
        seq_len
    );


    // ========================================================
    // Device memory
    // ========================================================

    float* d_scores =
        nullptr;


    float* d_weights =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_scores
            ),
            matrix_size
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_weights
            ),
            matrix_size
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_scores,
            h_scores.data(),
            matrix_size
            *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Reference output
    //
    // Current project baseline:
    //
    // block_size = 256
    // ========================================================

    constexpr int baseline_block_size =
        256;


    launch_attention_softmax(
        d_scores,
        d_weights,
        seq_len,
        baseline_block_size
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_reference.data(),
            d_weights,
            matrix_size
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // Benchmark configurations
    // ========================================================

    std::vector<BenchmarkResult>
        results;


    float baseline_time =
        0.0f;


    for(
        int block_size :
        block_sizes
    )
    {
        const float average_ms =
            benchmark_config(
                d_scores,
                d_weights,
                seq_len,
                block_size,
                warmup_runs,
                benchmark_runs
            );


        CUDA_CHECK(
            cudaMemcpy(
                h_output.data(),
                d_weights,
                matrix_size
                *
                sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );


        const bool correct =
            compare_outputs(
                h_reference,
                h_output,
                1e-5f
            );


        BenchmarkResult result;


        result.seq_len =
            seq_len;


        result.block_size =
            block_size;


        result.average_ms =
            average_ms;


        result.correct =
            correct;


        results.push_back(
            result
        );


        if(
            block_size
            ==
            baseline_block_size
        )
        {
            baseline_time =
                average_ms;
        }
    }


    if(
        baseline_time
        <=
        0.0f
    )
    {
        throw std::runtime_error(
            "256-thread baseline was not benchmarked"
        );
    }


    for(
        BenchmarkResult& result :
        results
    )
    {
        result.relative_to_256 =
            baseline_time
            /
            result.average_ms;
    }


    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(
        cudaFree(
            d_scores
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_weights
        )
    );


    return results;
}


// ============================================================
// Print results
// ============================================================

void print_results(
    const std::vector<BenchmarkResult>& results
)
{
    std::cout
        << std::left
        << std::setw(10)
        << "SeqLen"

        << std::setw(14)
        << "BlockSize"

        << std::setw(16)
        << "Time(ms)"

        << std::setw(18)
        << "vs256"

        << std::setw(12)
        << "Correct"

        << std::endl;


    std::cout
        << std::string(
            70,
            '-'
        )
        << std::endl;


    for(
        const BenchmarkResult& result :
        results
    )
    {
        std::cout
            << std::left

            << std::setw(10)
            << result.seq_len

            << std::setw(14)
            << result.block_size

            << std::setw(16)
            << std::fixed
            << std::setprecision(6)
            << result.average_ms

            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << result.relative_to_256

            << std::setw(12)
            << (
                result.correct
                ?
                "PASS"
                :
                "FAIL"
            )

            << std::endl;
    }
}


// ============================================================
// Print best block size per sequence length
// ============================================================

void print_best_configs(
    const std::vector<BenchmarkResult>& all_results,
    const std::vector<int>& sequence_lengths
)
{
    std::cout
        << "\nBest Block Size Per Sequence Length"
        << std::endl;


    std::cout
        << std::left

        << std::setw(10)
        << "SeqLen"

        << std::setw(14)
        << "BestBlock"

        << std::setw(16)
        << "Time(ms)"

        << std::setw(18)
        << "Speedup"

        << std::endl;


    std::cout
        << std::string(
            60,
            '-'
        )
        << std::endl;


    for(
        int seq_len :
        sequence_lengths
    )
    {
        const BenchmarkResult* best =
            nullptr;


        for(
            const BenchmarkResult& result :
            all_results
        )
        {
            if(
                result.seq_len
                !=
                seq_len
            )
            {
                continue;
            }


            if(
                !result.correct
            )
            {
                continue;
            }


            if(
                best == nullptr
                ||
                result.average_ms
                <
                best->average_ms
            )
            {
                best =
                    &result;
            }
        }


        if(
            best == nullptr
        )
        {
            continue;
        }


        std::cout
            << std::left

            << std::setw(10)
            << seq_len

            << std::setw(14)
            << best->block_size

            << std::setw(16)
            << std::fixed
            << std::setprecision(6)
            << best->average_ms

            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << best->relative_to_256

            << std::endl;
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        constexpr int warmup_runs =
            10;


        constexpr int benchmark_runs =
            100;


        const std::vector<int>
            sequence_lengths =
        {
            128,
            256,
            512,
            1024,
            2048
        };


        const std::vector<int>
            block_sizes =
        {
            64,
            128,
            256,
            512
        };


        std::vector<BenchmarkResult>
            all_results;


        std::cout
            << "\nAttentionSoftmax Block Size Benchmark"
            << std::endl;


        std::cout
            << "Warmup Runs: "
            << warmup_runs
            << std::endl;


        std::cout
            << "Benchmark Runs: "
            << benchmark_runs
            << std::endl;


        std::cout
            << "Baseline Block Size: 256"
            << std::endl;


        std::cout
            << std::endl;


        for(
            int seq_len :
            sequence_lengths
        )
        {
            std::vector<BenchmarkResult>
                results =
                    run_sequence_length(
                        seq_len,
                        block_sizes,
                        warmup_runs,
                        benchmark_runs
                    );


            print_results(
                results
            );


            std::cout
                << std::endl;


            all_results.insert(
                all_results.end(),
                results.begin(),
                results.end()
            );
        }


        print_best_configs(
            all_results,
            sequence_lengths
        );


        return 0;
    }
    catch(
        const std::exception& e
    )
    {
        std::cerr
            << "Benchmark exception: "
            << e.what()
            << std::endl;


        return 1;
    }
}