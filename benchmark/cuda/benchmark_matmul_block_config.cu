#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
// Tunable naive MatMul kernel
//
// A: [M, K]
// B: [K, N]
// C: [M, N]
//
// This intentionally keeps the SAME mathematical implementation
// as the current project MatMul kernel.
//
// The only experimental variable is:
//
//     blockDim.x
//     blockDim.y
//
// This allows us to isolate block-shape performance.
// ============================================================

__global__
void matmul_tuning_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
)
{
    const int col =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    const int row =
        blockIdx.y * blockDim.y
        + threadIdx.y;


    if(row >= M || col >= N)
        return;


    float sum = 0.0f;


    for(int k = 0; k < K; k++)
    {
        sum +=
            A[row * K + k]
            *
            B[k * N + col];
    }


    C[row * N + col] =
        sum;
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
// Launch helper
// ============================================================

void launch_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
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
            N
            + static_cast<int>(block.x)
            - 1
        )
        / static_cast<int>(block.x),

        (
            M
            + static_cast<int>(block.y)
            - 1
        )
        / static_cast<int>(block.y)
    );


    matmul_tuning_kernel<<<
        grid,
        block,
        0,
        stream
    >>>(
        A,
        B,
        C,
        M,
        K,
        N
    );
}


// ============================================================
// Deterministic input initialization
// ============================================================

void initialize_input(
    std::vector<float>& data,
    int seed
)
{
    for(std::size_t i = 0; i < data.size(); i++)
    {
        const int value =
            (
                static_cast<int>(i)
                + seed
            )
            % 101
            - 50;


        data[i] =
            static_cast<float>(value)
            * 0.01f;
    }
}


// ============================================================
// Output comparison
// ============================================================

bool compare_outputs(
    const std::vector<float>& reference,
    const std::vector<float>& output,
    float tolerance
)
{
    if(reference.size() != output.size())
        return false;


    for(std::size_t i = 0; i < reference.size(); i++)
    {
        const float diff =
            std::fabs(
                reference[i]
                - output[i]
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
// Benchmark result
// ============================================================

struct BenchmarkResult
{
    int seq_len = 0;

    int block_x = 0;

    int block_y = 0;

    int threads_per_block = 0;

    float average_ms = 0.0f;

    float relative_to_16x16 = 0.0f;

    bool correct = false;
};


// ============================================================
// Benchmark one block configuration
// ============================================================

float benchmark_config(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M,
    int K,
    int N,
    const BlockConfig& config,
    int warmup_runs,
    int benchmark_runs
)
{
    // ----------------------------
    // Warmup
    // ----------------------------

    for(int i = 0; i < warmup_runs; i++)
    {
        launch_matmul(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N,
            config
        );
    }


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ----------------------------
    // CUDA events
    // ----------------------------

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


    for(int i = 0; i < benchmark_runs; i++)
    {
        launch_matmul(
            d_A,
            d_B,
            d_C,
            M,
            K,
            N,
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
//
// Attention output MatMul shape:
//
// weights: [seq_len, seq_len]
// V:       [seq_len, head_dim]
// output:  [seq_len, head_dim]
//
// Therefore:
//
// M = seq_len
// K = seq_len
// N = head_dim
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
    const int M =
        seq_len;

    const int K =
        seq_len;

    const int N =
        head_dim;


    const std::size_t a_size =
        static_cast<std::size_t>(M)
        *
        static_cast<std::size_t>(K);


    const std::size_t b_size =
        static_cast<std::size_t>(K)
        *
        static_cast<std::size_t>(N);


    const std::size_t c_size =
        static_cast<std::size_t>(M)
        *
        static_cast<std::size_t>(N);


    // ----------------------------
    // Host memory
    // ----------------------------

    std::vector<float> h_A(
        a_size
    );


    std::vector<float> h_B(
        b_size
    );


    std::vector<float> h_reference(
        c_size
    );


    std::vector<float> h_output(
        c_size
    );


    initialize_input(
        h_A,
        1
    );


    initialize_input(
        h_B,
        17
    );


    // ----------------------------
    // Device memory
    // ----------------------------

    float* d_A =
        nullptr;

    float* d_B =
        nullptr;

    float* d_C =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_A
            ),
            a_size
            * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_B
            ),
            b_size
            * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_C
            ),
            c_size
            * sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_A,
            h_A.data(),
            a_size
            * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_B,
            h_B.data(),
            b_size
            * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Correctness reference
    //
    // Use current project baseline:
    //
    // 16 x 16
    // ========================================================

    const BlockConfig baseline_config =
    {
        16,
        16
    };


    launch_matmul(
        d_A,
        d_B,
        d_C,
        M,
        K,
        N,
        baseline_config
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_reference.data(),
            d_C,
            c_size
            * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // Benchmark every configuration
    // ========================================================

    std::vector<BenchmarkResult> results;


    float baseline_time =
        0.0f;


    for(const BlockConfig& config : configs)
    {
        const float average_ms =
            benchmark_config(
                d_A,
                d_B,
                d_C,
                M,
                K,
                N,
                config,
                warmup_runs,
                benchmark_runs
            );


        CUDA_CHECK(
            cudaMemcpy(
                h_output.data(),
                d_C,
                c_size
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


    if(baseline_time <= 0.0f)
    {
        throw std::runtime_error(
            "16x16 baseline configuration was not benchmarked"
        );
    }


    for(BenchmarkResult& result : results)
    {
        result.relative_to_16x16 =
            baseline_time
            /
            result.average_ms;
    }


    // ----------------------------
    // Cleanup
    // ----------------------------

    CUDA_CHECK(
        cudaFree(
            d_A
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_B
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_C
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


    for(const BenchmarkResult& result : results)
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
                ? "PASS"
                : "FAIL"
            )
            << std::endl;
    }
}


// ============================================================
// Find and print best configuration per sequence length
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


    for(int seq_len : sequence_lengths)
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


            if(!result.correct)
                continue;


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


        if(best == nullptr)
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


        const std::vector<int> sequence_lengths =
        {
            128,
            256,
            512,
            1024,
            2048
        };


        // Current baseline:
        //
        //     16 x 16
        //
        // Experimental alternatives:
        //
        //     8 x 8
        //     32 x 8
        //     8 x 32
        //
        const std::vector<BlockConfig> configs =
        {
            {16, 16},
            {8, 8},
            {32, 8},
            {8, 32}
        };


        std::vector<BenchmarkResult>
            all_results;


        std::cout
            << "\nMatMul Block Configuration Benchmark"
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
            << "A = [seq_len, seq_len]"
            << std::endl;


        std::cout
            << "B = [seq_len, 64]"
            << std::endl;


        std::cout
            << "C = [seq_len, 64]"
            << std::endl;


        std::cout
            << std::endl;


        for(int seq_len : sequence_lengths)
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
    catch(const std::exception& e)
    {
        std::cerr
            << "Benchmark exception: "
            << e.what()
            << std::endl;


        return 1;
    }
}