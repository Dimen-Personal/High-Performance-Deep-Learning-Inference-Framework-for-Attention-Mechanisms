#include "cuda/cuda_ops.cuh"
#include "cuda/flash_attention.cuh"

#include <cuda_runtime.h>

#include <algorithm>
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
                                                                             \
        if(error != cudaSuccess)                                             \
        {                                                                    \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ")                                  \
                + cudaGetErrorString(error)                                  \
            );                                                               \
        }                                                                    \
    } while(false)


// ============================================================
// Configuration
// ============================================================

constexpr int WARMUP_RUNS =
    10;


constexpr int BENCHMARK_RUNS =
    100;


constexpr int HEAD_DIM =
    64;


// ============================================================
// Result
// ============================================================

struct BenchmarkResult
{
    int seq_len =
        0;


    float baseline_ms =
        0.0f;


    float naive_fused_ms =
        0.0f;


    float tiled_flash_ms =
        0.0f;


    float naive_speedup =
        0.0f;


    float tiled_speedup =
        0.0f;


    double temporary_memory_mb =
        0.0;


    double minimum_intermediate_traffic_mb =
        0.0;


    double current_baseline_logical_traffic_mb =
        0.0;
};


// ============================================================
// Deterministic initialization
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
                +
                seed
            )
            %
            101
            -
            50;


        data[i] =
            static_cast<float>(
                value
            )
            *
            0.01f;
    }
}


// ============================================================
// Baseline Attention
//
// AttentionScores
//      ↓
// Softmax
//      ↓
// MatMul(weights, V)
// ============================================================

void launch_baseline_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* scores,
    float* weights,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream
)
{
    cuda_attention_scores(
        Q,
        K,
        scores,
        seq_len,
        head_dim,
        stream
    );


    cuda_attention_softmax(
        scores,
        weights,
        seq_len,
        stream
    );


    cuda_matmul(
        weights,
        V,
        output,
        seq_len,
        seq_len,
        head_dim,
        stream
    );
}


// ============================================================
// Benchmark baseline
// ============================================================

float benchmark_baseline(
    const float* Q,
    const float* K,
    const float* V,
    float* scores,
    float* weights,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream
)
{
    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for(
        int i = 0;
        i < WARMUP_RUNS;
        i++
    )
    {
        launch_baseline_attention(
            Q,
            K,
            V,
            scores,
            weights,
            output,
            seq_len,
            head_dim,
            stream
        );
    }


    CUDA_CHECK(
        cudaStreamSynchronize(
            stream
        )
    );


    // --------------------------------------------------------
    // Events
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
            start,
            stream
        )
    );


    for(
        int i = 0;
        i < BENCHMARK_RUNS;
        i++
    )
    {
        launch_baseline_attention(
            Q,
            K,
            V,
            scores,
            weights,
            output,
            seq_len,
            head_dim,
            stream
        );
    }


    CUDA_CHECK(
        cudaEventRecord(
            stop,
            stream
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
            BENCHMARK_RUNS
        );
}


// ============================================================
// Benchmark naive fused
// ============================================================

float benchmark_naive_fused(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream
)
{
    for(
        int i = 0;
        i < WARMUP_RUNS;
        i++
    )
    {
        cuda_flash_attention(
            Q,
            K,
            V,
            output,
            seq_len,
            head_dim,
            stream
        );
    }


    CUDA_CHECK(
        cudaStreamSynchronize(
            stream
        )
    );


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
            start,
            stream
        )
    );


    for(
        int i = 0;
        i < BENCHMARK_RUNS;
        i++
    )
    {
        cuda_flash_attention(
            Q,
            K,
            V,
            output,
            seq_len,
            head_dim,
            stream
        );
    }


    CUDA_CHECK(
        cudaEventRecord(
            stop,
            stream
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
            BENCHMARK_RUNS
        );
}


// ============================================================
// Benchmark tiled Flash Attention
// ============================================================

float benchmark_tiled_flash(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream
)
{
    for(
        int i = 0;
        i < WARMUP_RUNS;
        i++
    )
    {
        cuda_flash_attention_tiled(
            Q,
            K,
            V,
            output,
            seq_len,
            head_dim,
            stream
        );
    }


    CUDA_CHECK(
        cudaStreamSynchronize(
            stream
        )
    );


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
            start,
            stream
        )
    );


    for(
        int i = 0;
        i < BENCHMARK_RUNS;
        i++
    )
    {
        cuda_flash_attention_tiled(
            Q,
            K,
            V,
            output,
            seq_len,
            head_dim,
            stream
        );
    }


    CUDA_CHECK(
        cudaEventRecord(
            stop,
            stream
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
            BENCHMARK_RUNS
        );
}


// ============================================================
// Run one sequence length
// ============================================================

BenchmarkResult run_sequence_length(
    int seq_len
)
{
    const std::size_t qkv_elements =
        static_cast<std::size_t>(
            seq_len
        )
        *
        HEAD_DIM;


    const std::size_t attention_elements =
        static_cast<std::size_t>(
            seq_len
        )
        *
        seq_len;


    // ========================================================
    // Host input
    // ========================================================

    std::vector<float> h_Q(
        qkv_elements
    );


    std::vector<float> h_K(
        qkv_elements
    );


    std::vector<float> h_V(
        qkv_elements
    );


    initialize_input(
        h_Q,
        1
    );


    initialize_input(
        h_K,
        17
    );


    initialize_input(
        h_V,
        37
    );


    // ========================================================
    // Device buffers
    // ========================================================

    float* d_Q =
        nullptr;


    float* d_K =
        nullptr;


    float* d_V =
        nullptr;


    float* d_scores =
        nullptr;


    float* d_weights =
        nullptr;


    float* d_output =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_Q,
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_K,
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_V,
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_scores,
            attention_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_weights,
            attention_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_Q,
            h_Q.data(),
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_K,
            h_K.data(),
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_V,
            h_V.data(),
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Stream
    // ========================================================

    cudaStream_t stream;


    CUDA_CHECK(
        cudaStreamCreate(
            &stream
        )
    );


    // ========================================================
    // Benchmark
    // ========================================================

    const float baseline_ms =
        benchmark_baseline(
            d_Q,
            d_K,
            d_V,
            d_scores,
            d_weights,
            d_output,
            seq_len,
            HEAD_DIM,
            stream
        );


    const float naive_fused_ms =
        benchmark_naive_fused(
            d_Q,
            d_K,
            d_V,
            d_output,
            seq_len,
            HEAD_DIM,
            stream
        );


    const float tiled_flash_ms =
        benchmark_tiled_flash(
            d_Q,
            d_K,
            d_V,
            d_output,
            seq_len,
            HEAD_DIM,
            stream
        );


    // ========================================================
    // Memory analysis
    //
    // Standard Attention temporary allocation:
    //
    // scores  = N × N floats
    // weights = N × N floats
    //
    // total = 2 × N² × sizeof(float)
    // ========================================================

    const double temporary_bytes =
        2.0
        *
        static_cast<double>(
            attention_elements
        )
        *
        sizeof(float);


    // ========================================================
    // Minimum algorithmic intermediate traffic:
    //
    // scores:
    //     write + read
    //
    // weights:
    //     write + read
    //
    // = 4 × N² floats
    // ========================================================

    const double minimum_traffic_bytes =
        4.0
        *
        static_cast<double>(
            attention_elements
        )
        *
        sizeof(float);


    // ========================================================
    // Current baseline kernel logical intermediate traffic:
    //
    // scores:
    //     write
    //     read
    //
    // softmax weights:
    //     write exp
    //     read for normalization
    //     write normalized value
    //
    // final MatMul:
    //     read weights
    //
    // = 6 × N² floats
    //
    // This is logical traffic, not guaranteed physical DRAM
    // transactions because caches may absorb some accesses.
    // ========================================================

    const double current_baseline_traffic_bytes =
        6.0
        *
        static_cast<double>(
            attention_elements
        )
        *
        sizeof(float);


    BenchmarkResult result;


    result.seq_len =
        seq_len;


    result.baseline_ms =
        baseline_ms;


    result.naive_fused_ms =
        naive_fused_ms;


    result.tiled_flash_ms =
        tiled_flash_ms;


    result.naive_speedup =
        baseline_ms
        /
        naive_fused_ms;


    result.tiled_speedup =
        baseline_ms
        /
        tiled_flash_ms;


    result.temporary_memory_mb =
        temporary_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.minimum_intermediate_traffic_mb =
        minimum_traffic_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.current_baseline_logical_traffic_mb =
        current_baseline_traffic_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(
        cudaStreamDestroy(
            stream
        )
    );


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
            d_V
        )
    );


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


    CUDA_CHECK(
        cudaFree(
            d_output
        )
    );


    return result;
}


// ============================================================
// Print performance
// ============================================================

void print_performance_results(
    const std::vector<BenchmarkResult>& results
)
{
    std::cout
        << "\nPerformance Results"
        << std::endl;


    std::cout
        << std::left

        << std::setw(10)
        << "SeqLen"

        << std::setw(18)
        << "Baseline(ms)"

        << std::setw(18)
        << "NaiveFused(ms)"

        << std::setw(18)
        << "TiledFlash(ms)"

        << std::setw(16)
        << "NaiveSpeedup"

        << std::setw(16)
        << "FlashSpeedup"

        << std::endl;


    std::cout
        << std::string(
            96,
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

            << std::setw(18)
            << std::fixed
            << std::setprecision(6)
            << result.baseline_ms

            << std::setw(18)
            << std::fixed
            << std::setprecision(6)
            << result.naive_fused_ms

            << std::setw(18)
            << std::fixed
            << std::setprecision(6)
            << result.tiled_flash_ms

            << std::setw(16)
            << std::fixed
            << std::setprecision(4)
            << result.naive_speedup

            << std::setw(16)
            << std::fixed
            << std::setprecision(4)
            << result.tiled_speedup

            << std::endl;
    }
}


// ============================================================
// Print memory results
// ============================================================

void print_memory_results(
    const std::vector<BenchmarkResult>& results
)
{
    std::cout
        << "\nIntermediate Memory / Traffic"
        << std::endl;


    std::cout
        << std::left

        << std::setw(10)
        << "SeqLen"

        << std::setw(22)
        << "TempMemory(MB)"

        << std::setw(24)
        << "MinTraffic(MB)"

        << std::setw(26)
        << "CurrentLogical(MB)"

        << std::endl;


    std::cout
        << std::string(
            82,
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

            << std::setw(22)
            << std::fixed
            << std::setprecision(4)
            << result.temporary_memory_mb

            << std::setw(24)
            << std::fixed
            << std::setprecision(4)
            << result.minimum_intermediate_traffic_mb

            << std::setw(26)
            << std::fixed
            << std::setprecision(4)
            << result.current_baseline_logical_traffic_mb

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
        const std::vector<int>
            sequence_lengths =
        {
            128,
            256,
            512,
            1024,
            2048
        };


        std::vector<BenchmarkResult>
            results;


        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "Flash Attention Performance Benchmark"
            << std::endl;


        std::cout
            << "============================================"
            << std::endl;


        std::cout
            << "Head Dimension: "
            << HEAD_DIM
            << std::endl;


        std::cout
            << "Warmup Runs: "
            << WARMUP_RUNS
            << std::endl;


        std::cout
            << "Benchmark Runs: "
            << BENCHMARK_RUNS
            << std::endl;


        std::cout
            << std::endl;


        for(
            int seq_len :
            sequence_lengths
        )
        {
            std::cout
                << "Running seq_len = "
                << seq_len
                << "..."
                << std::endl;


            results.push_back(
                run_sequence_length(
                    seq_len
                )
            );
        }


        print_performance_results(
            results
        );


        print_memory_results(
            results
        );


        std::cout
            << "\nNotes:"
            << std::endl;


        std::cout
            << "TempMemory = scores[N,N] + weights[N,N]"
            << std::endl;


        std::cout
            << "MinTraffic = theoretical minimum intermediate materialization traffic"
            << std::endl;


        std::cout
            << "CurrentLogical = logical intermediate accesses in current baseline kernels"
            << std::endl;


        std::cout
            << "Actual DRAM traffic may differ because of GPU caches."
            << std::endl;


        return 0;
    }
    catch(
        const std::exception& exception
    )
    {
        std::cerr
            << "Benchmark exception: "
            << exception.what()
            << std::endl;


        return 1;
    }
}