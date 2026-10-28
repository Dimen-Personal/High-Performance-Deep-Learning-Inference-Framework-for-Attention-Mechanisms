#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
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
// Tunable AttentionScores kernel
//
// Q:      [seq_len, head_dim]
// K:      [seq_len, head_dim]
// scores: [seq_len, seq_len]
//
// scores[row][col] =
//     dot(Q[row], K[col]) / sqrt(head_dim)
//
// Causal mask:
//
// col > row
// → -INFINITY
//
// The mathematical implementation is intentionally kept
// equivalent to the current project AttentionScores kernel.
//
// Only blockDim.x / blockDim.y are changed.
// ============================================================

__global__
void attention_scores_tuning_kernel(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim
)
{
    const int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    const int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;


    if(
        row >= seq_len
        ||
        col >= seq_len
    )
    {
        return;
    }


    // Causal mask
    if(col > row)
    {
        scores[
            row * seq_len
            + col
        ] = -INFINITY;

        return;
    }


    float sum =
        0.0f;


    for(
        int k = 0;
        k < head_dim;
        k++
    )
    {
        sum +=
            Q[
                row * head_dim
                + k
            ]
            *
            K[
                col * head_dim
                + k
            ];
    }


    const float scale =
        1.0f
        /
        sqrtf(
            static_cast<float>(
                head_dim
            )
        );


    scores[
        row * seq_len
        + col
    ] =
        sum * scale;
}


// ============================================================
// Block configuration
// ============================================================

struct BlockConfig
{
    int x;
    int y;
};


// ============================================================
// Benchmark result
// ============================================================

struct BenchmarkResult
{
    int seq_len =
        0;

    int block_x =
        0;

    int block_y =
        0;

    int threads_per_block =
        0;

    float average_ms =
        0.0f;

    float relative_to_16x16 =
        0.0f;

    bool correct =
        false;
};


// ============================================================
// Kernel launcher
// ============================================================

void launch_attention_scores(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim,
    const BlockConfig& config,
    cudaStream_t stream = nullptr
)
{
    dim3 block(
        config.x,
        config.y
    );


    dim3 grid(
        (
            seq_len
            + static_cast<int>(block.x)
            - 1
        )
        /
        static_cast<int>(
            block.x
        ),

        (
            seq_len
            + static_cast<int>(block.y)
            - 1
        )
        /
        static_cast<int>(
            block.y
        )
    );


    attention_scores_tuning_kernel<<<
        grid,
        block,
        0,
        stream
    >>>(
        Q,
        K,
        scores,
        seq_len,
        head_dim
    );
}


// ============================================================
// Deterministic input
// ============================================================

void initialize_input(
    std::vector<float>& data,
    int seed
)
{
    for(
        std::size_t i = 0;
        i < data.size();
        i++
    )
    {
        const int value =
            (
                static_cast<int>(i)
                + seed
            )
            % 101
            - 50;


        data[i] =
            static_cast<float>(
                value
            )
            *
            0.01f;
    }
}


// ============================================================
// Compare outputs
//
// Need special handling for -INFINITY from causal mask.
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
        const float ref =
            reference[i];

        const float out =
            output[i];


        // Both masked values
        if(
            std::isinf(ref)
            &&
            std::isinf(out)
            &&
            ref < 0.0f
            &&
            out < 0.0f
        )
        {
            continue;
        }


        const float diff =
            std::fabs(
                ref
                -
                out
            );


        if(
            diff
            >
            tolerance
        )
        {
            std::cerr
                << "Mismatch at index "
                << i
                << ": reference = "
                << ref
                << ", output = "
                << out
                << ", diff = "
                << diff
                << std::endl;


            return false;
        }
    }


    return true;
}


// ============================================================
// Benchmark one block configuration
// ============================================================

float benchmark_config(
    const float* d_Q,
    const float* d_K,
    float* d_scores,
    int seq_len,
    int head_dim,
    const BlockConfig& config,
    int warmup_runs,
    int benchmark_runs
)
{
    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for(
        int i = 0;
        i < warmup_runs;
        i++
    )
    {
        launch_attention_scores(
            d_Q,
            d_K,
            d_scores,
            seq_len,
            head_dim,
            config
        );
    }


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------

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
        launch_attention_scores(
            d_Q,
            d_K,
            d_scores,
            seq_len,
            head_dim,
            config
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
    int head_dim,
    const std::vector<BlockConfig>& configs,
    int warmup_runs,
    int benchmark_runs
)
{
    const std::size_t qk_size =
        static_cast<std::size_t>(
            seq_len
        )
        *
        static_cast<std::size_t>(
            head_dim
        );


    const std::size_t scores_size =
        static_cast<std::size_t>(
            seq_len
        )
        *
        static_cast<std::size_t>(
            seq_len
        );


    // --------------------------------------------------------
    // Host memory
    // --------------------------------------------------------

    std::vector<float> h_Q(
        qk_size
    );


    std::vector<float> h_K(
        qk_size
    );


    std::vector<float> h_reference(
        scores_size
    );


    std::vector<float> h_output(
        scores_size
    );


    initialize_input(
        h_Q,
        1
    );


    initialize_input(
        h_K,
        17
    );


    // --------------------------------------------------------
    // Device memory
    // --------------------------------------------------------

    float* d_Q =
        nullptr;

    float* d_K =
        nullptr;

    float* d_scores =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_Q
            ),
            qk_size
            * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_K
            ),
            qk_size
            * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_scores
            ),
            scores_size
            * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_Q,
            h_Q.data(),
            qk_size
            * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_K,
            h_K.data(),
            qk_size
            * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Correctness reference
    //
    // Current project baseline:
    //
    // 16 × 16
    // ========================================================

    const BlockConfig baseline_config =
    {
        16,
        16
    };


    launch_attention_scores(
        d_Q,
        d_K,
        d_scores,
        seq_len,
        head_dim,
        baseline_config
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_reference.data(),
            d_scores,
            scores_size
            * sizeof(float),
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
        const BlockConfig& config :
        configs
    )
    {
        const float average_ms =
            benchmark_config(
                d_Q,
                d_K,
                d_scores,
                seq_len,
                head_dim,
                config,
                warmup_runs,
                benchmark_runs
            );


        CUDA_CHECK(
            cudaMemcpy(
                h_output.data(),
                d_scores,
                scores_size
                * sizeof(float),
                cudaMemcpyDeviceToHost
            )
        );


        const bool correct =
            compare_outputs(
                h_reference,
                h_output,
                1e-3f
            );


        BenchmarkResult result;


        result.seq_len =
            seq_len;


        result.block_x =
            config.x;


        result.block_y =
            config.y;


        result.threads_per_block =
            config.x
            *
            config.y;


        result.average_ms =
            average_ms;


        result.correct =
            correct;


        results.push_back(
            result
        );


        if(
            config.x == 16
            &&
            config.y == 16
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
            "16x16 baseline configuration was not benchmarked"
        );
    }


    for(
        BenchmarkResult& result :
        results
    )
    {
        result.relative_to_16x16 =
            baseline_time
            /
            result.average_ms;
    }


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaFree(
            d_Q
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_K
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_scores
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
        << std::setw(12)
        << "Block"
        << std::setw(12)
        << "Threads"
        << std::setw(16)
        << "Time(ms)"
        << std::setw(18)
        << "vs16x16"
        << std::setw(12)
        << "Correct"
        << std::endl;


    std::cout
        << std::string(
            80,
            '-'
        )
        << std::endl;


    for(
        const BenchmarkResult& result :
        results
    )
    {
        const std::string block_name =
            std::to_string(
                result.block_x
            )
            +
            "x"
            +
            std::to_string(
                result.block_y
            );


        std::cout
            << std::left
            << std::setw(10)
            << result.seq_len
            << std::setw(12)
            << block_name
            << std::setw(12)
            << result.threads_per_block
            << std::setw(16)
            << std::fixed
            << std::setprecision(6)
            << result.average_ms
            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << result.relative_to_16x16
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
// Print best configuration per sequence length
// ============================================================

void print_best_configs(
    const std::vector<BenchmarkResult>& all_results,
    const std::vector<int>& sequence_lengths
)
{
    std::cout
        << "\nBest Configuration Per Sequence Length"
        << std::endl;


    std::cout
        << std::left
        << std::setw(10)
        << "SeqLen"
        << std::setw(12)
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
            std::cout
                << seq_len
                << " : no valid result"
                << std::endl;

            continue;
        }


        const std::string block_name =
            std::to_string(
                best->block_x
            )
            +
            "x"
            +
            std::to_string(
                best->block_y
            );


        std::cout
            << std::left
            << std::setw(10)
            << seq_len
            << std::setw(12)
            << block_name
            << std::setw(16)
            << std::fixed
            << std::setprecision(6)
            << best->average_ms
            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << best->relative_to_16x16
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
        constexpr int head_dim =
            64;


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


        const std::vector<BlockConfig>
            configs =
        {
            {16, 16},
            {8, 8},
            {32, 8},
            {8, 32}
        };


        std::vector<BenchmarkResult>
            all_results;


        std::cout
            << "\nAttentionScores Block Configuration Benchmark"
            << std::endl;


        std::cout
            << "Head Dimension: "
            << head_dim
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
            << "\nShape:"
            << std::endl;


        std::cout
            << "Q = [seq_len, 64]"
            << std::endl;


        std::cout
            << "K = [seq_len, 64]"
            << std::endl;


        std::cout
            << "Scores = [seq_len, seq_len]"
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
                        head_dim,
                        configs,
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