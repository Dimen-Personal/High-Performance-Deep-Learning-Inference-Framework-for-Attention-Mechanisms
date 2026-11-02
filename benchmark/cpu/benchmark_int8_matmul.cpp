#include "quantization/int8_matmul.h"
#include "quantization/quantization.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>


// ============================================================
// Benchmark configuration
// ============================================================

constexpr int WARMUP_RUNS =
    3;

constexpr int BENCHMARK_RUNS =
    10;


// ============================================================
// Benchmark result
// ============================================================

struct BenchmarkResult
{
    int matrix_size = 0;

    double fp32_ms = 0.0;

    double int8_kernel_ms = 0.0;

    double int8_end_to_end_ms = 0.0;

    double kernel_speedup = 0.0;

    double end_to_end_speedup = 0.0;

    double mae = 0.0;

    float max_error = 0.0f;

    std::size_t fp32_input_bytes = 0;

    std::size_t int8_input_bytes = 0;

    double memory_reduction = 0.0;
};


// ============================================================
// Deterministic matrix initialization
// ============================================================

void initialize_matrix(
    std::vector<float>& matrix,
    int seed
)
{
    std::mt19937 generator(
        seed
    );


    std::uniform_real_distribution<float>
        distribution(
            -1.0f,
            1.0f
        );


    for(
        float& value :
        matrix
    )
    {
        value =
            distribution(
                generator
            );
    }
}


// ============================================================
// FP32 MatMul
//
// A: [M, K]
// B: [K, N]
// C: [M, N]
// ============================================================

void fp32_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
)
{
    for(int i = 0; i < M; i++)
    {
        for(int j = 0; j < N; j++)
        {
            float sum = 0.0f;


            for(int k = 0; k < K; k++)
                sum +=A[i * K + k] * B[k * N + j];
            


            C[i * N+ j] = sum;
        }
    }
}


// ============================================================
// Error metrics
// ============================================================

void calculate_error(
    const std::vector<float>& reference,
    const std::vector<float>& output,
    double& mae,
    float& max_error
)
{
    if(
        reference.size()
        !=
        output.size()
    )
    {
        throw std::runtime_error(
            "Output size mismatch."
        );
    }


    double total_error =
        0.0;


    max_error =
        0.0f;


    for(std::size_t i = 0; i < reference.size(); i++)
    {
        const float error = std::fabs(reference[i] - output[i]);


        total_error += static_cast<double>(error);


        max_error = std::max(max_error, error);
    }


    mae = total_error / static_cast<double>(reference.size());
}


// ============================================================
// FP32 benchmark
// ============================================================

double benchmark_fp32(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
)
{
    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for(
        int run = 0;
        run < WARMUP_RUNS;
        run++
    )
    {
        fp32_matmul(
            A,
            B,
            C,
            M,
            K,
            N
        );
    }


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    const auto start =
        std::chrono::steady_clock::now();


    for(
        int run = 0;
        run < BENCHMARK_RUNS;
        run++
    )
    {
        fp32_matmul(
            A,
            B,
            C,
            M,
            K,
            N
        );
    }


    const auto stop =
        std::chrono::steady_clock::now();


    const double total_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            stop - start
        ).count();


    return
        total_ms
        /
        static_cast<double>(
            BENCHMARK_RUNS
        );
}


// ============================================================
// INT8 kernel-only benchmark
//
// Quantization happens outside timing.
// ============================================================

double benchmark_int8_kernel(
    const QuantizedTensor& A_quantized,
    const QuantizedTensor& B_quantized,
    float* C,
    int M,
    int K,
    int N
)
{
    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for(
        int run = 0;
        run < WARMUP_RUNS;
        run++
    )
    {
        int8_matmul(
            A_quantized.data.data(),
            B_quantized.data.data(),
            C,
            M,
            K,
            N,
            A_quantized.params,
            B_quantized.params
        );
    }


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    const auto start =
        std::chrono::steady_clock::now();


    for(
        int run = 0;
        run < BENCHMARK_RUNS;
        run++
    )
    {
        int8_matmul(
            A_quantized.data.data(),
            B_quantized.data.data(),
            C,
            M,
            K,
            N,
            A_quantized.params,
            B_quantized.params
        );
    }


    const auto stop =
        std::chrono::steady_clock::now();


    const double total_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            stop - start
        ).count();


    return
        total_ms
        /
        static_cast<double>(
            BENCHMARK_RUNS
        );
}


// ============================================================
// INT8 end-to-end benchmark
//
// This includes:
//
// FP32 A
//   ↓
// quantize
//
// FP32 B
//   ↓
// quantize
//
// INT8 MatMul
//
// This intentionally measures runtime quantization overhead.
// ============================================================

double benchmark_int8_end_to_end(
    const float* A,
    std::size_t A_size,
    const float* B,
    std::size_t B_size,
    float* C,
    int M,
    int K,
    int N
)
{
    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    for(
        int run = 0;
        run < WARMUP_RUNS;
        run++
    )
    {
        const QuantizedTensor A_quantized =
            quantize_fp32_to_int8(
                A,
                A_size
            );


        const QuantizedTensor B_quantized =
            quantize_fp32_to_int8(
                B,
                B_size
            );


        int8_matmul(
            A_quantized.data.data(),
            B_quantized.data.data(),
            C,
            M,
            K,
            N,
            A_quantized.params,
            B_quantized.params
        );
    }


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    const auto start =
        std::chrono::steady_clock::now();


    for(
        int run = 0;
        run < BENCHMARK_RUNS;
        run++
    )
    {
        const QuantizedTensor A_quantized =
            quantize_fp32_to_int8(
                A,
                A_size
            );


        const QuantizedTensor B_quantized =
            quantize_fp32_to_int8(
                B,
                B_size
            );


        int8_matmul(
            A_quantized.data.data(),
            B_quantized.data.data(),
            C,
            M,
            K,
            N,
            A_quantized.params,
            B_quantized.params
        );
    }


    const auto stop =
        std::chrono::steady_clock::now();


    const double total_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            stop - start
        ).count();


    return
        total_ms
        /
        static_cast<double>(
            BENCHMARK_RUNS
        );
}


// ============================================================
// Run one matrix size
// ============================================================

BenchmarkResult run_matrix_size(
    int matrix_size
)
{
    const int M =
        matrix_size;

    const int K =
        matrix_size;

    const int N =
        matrix_size;


    const std::size_t A_size =
        static_cast<std::size_t>(
            M
        )
        *
        K;


    const std::size_t B_size =
        static_cast<std::size_t>(
            K
        )
        *
        N;


    const std::size_t C_size =
        static_cast<std::size_t>(
            M
        )
        *
        N;


    // ========================================================
    // Allocate matrices
    // ========================================================

    std::vector<float> A(
        A_size
    );


    std::vector<float> B(
        B_size
    );


    std::vector<float> C_fp32(
        C_size
    );


    std::vector<float> C_int8(
        C_size
    );


    initialize_matrix(
        A,
        1
    );


    initialize_matrix(
        B,
        2
    );


    // ========================================================
    // Quantize once for kernel-only benchmark
    // ========================================================

    const QuantizedTensor A_quantized =
        quantize_fp32_to_int8(
            A.data(),
            A.size()
        );


    const QuantizedTensor B_quantized =
        quantize_fp32_to_int8(
            B.data(),
            B.size()
        );


    // ========================================================
    // FP32
    // ========================================================

    const double fp32_ms =
        benchmark_fp32(
            A.data(),
            B.data(),
            C_fp32.data(),
            M,
            K,
            N
        );


    // ========================================================
    // INT8 kernel-only
    // ========================================================

    const double int8_kernel_ms =
        benchmark_int8_kernel(
            A_quantized,
            B_quantized,
            C_int8.data(),
            M,
            K,
            N
        );


    // ========================================================
    // Numerical error
    // ========================================================

    double mae =
        0.0;


    float max_error =
        0.0f;


    calculate_error(
        C_fp32,
        C_int8,
        mae,
        max_error
    );


    // ========================================================
    // INT8 End-to-End
    // ========================================================

    const double int8_end_to_end_ms =
        benchmark_int8_end_to_end(
            A.data(),
            A.size(),
            B.data(),
            B.size(),
            C_int8.data(),
            M,
            K,
            N
        );


    // ========================================================
    // Memory
    // ========================================================

    const std::size_t fp32_input_bytes =
        (
            A_size
            +
            B_size
        )
        *
        sizeof(float);


    const std::size_t int8_input_bytes =
        (
            A_size
            +
            B_size
        )
        *
        sizeof(int8_t);


    const double memory_reduction =
        1.0
        -
        static_cast<double>(
            int8_input_bytes
        )
        /
        static_cast<double>(
            fp32_input_bytes
        );


    // ========================================================
    // Result
    // ========================================================

    BenchmarkResult result;


    result.matrix_size =
        matrix_size;


    result.fp32_ms =
        fp32_ms;


    result.int8_kernel_ms =
        int8_kernel_ms;


    result.int8_end_to_end_ms =
        int8_end_to_end_ms;


    result.kernel_speedup =
        fp32_ms
        /
        int8_kernel_ms;


    result.end_to_end_speedup =
        fp32_ms
        /
        int8_end_to_end_ms;


    result.mae =
        mae;


    result.max_error =
        max_error;


    result.fp32_input_bytes =
        fp32_input_bytes;


    result.int8_input_bytes =
        int8_input_bytes;


    result.memory_reduction =
        memory_reduction;


    return result;
}


// ============================================================
// Print result table
// ============================================================

void print_results(
    const std::vector<BenchmarkResult>& results
)
{
    std::cout
        << std::left

        << std::setw(10)
        << "Size"

        << std::setw(16)
        << "FP32(ms)"

        << std::setw(18)
        << "INT8Kernel(ms)"

        << std::setw(18)
        << "INT8E2E(ms)"

        << std::setw(16)
        << "KernelSpeedup"

        << std::setw(16)
        << "E2ESpeedup"

        << std::setw(14)
        << "MAE"

        << std::endl;


    std::cout
        << std::string(
            108,
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
            << result.matrix_size

            << std::setw(16)
            << std::fixed
            << std::setprecision(4)
            << result.fp32_ms

            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << result.int8_kernel_ms

            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << result.int8_end_to_end_ms

            << std::setw(16)
            << std::fixed
            << std::setprecision(4)
            << result.kernel_speedup

            << std::setw(16)
            << std::fixed
            << std::setprecision(4)
            << result.end_to_end_speedup

            << std::setw(14)
            << std::fixed
            << std::setprecision(6)
            << result.mae

            << std::endl;
    }
}


// ============================================================
// Print memory information
// ============================================================

void print_memory_results(
    const std::vector<BenchmarkResult>& results
)
{
    std::cout
        << "\nMemory Comparison"
        << std::endl;


    std::cout
        << std::left

        << std::setw(10)
        << "Size"

        << std::setw(18)
        << "FP32 Input"

        << std::setw(18)
        << "INT8 Input"

        << std::setw(18)
        << "Reduction"

        << std::endl;


    std::cout
        << std::string(
            64,
            '-'
        )
        << std::endl;


    for(
        const BenchmarkResult& result :
        results
    )
    {
        const double fp32_mb =
            static_cast<double>(
                result.fp32_input_bytes
            )
            /
            (
                1024.0
                *
                1024.0
            );


        const double int8_mb =
            static_cast<double>(
                result.int8_input_bytes
            )
            /
            (
                1024.0
                *
                1024.0
            );


        std::cout
            << std::left

            << std::setw(10)
            << result.matrix_size

            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << fp32_mb

            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << int8_mb

            << std::setw(18)
            << std::fixed
            << std::setprecision(2)
            << (
                result.memory_reduction
                *
                100.0
            )

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
        const std::vector<int> matrix_sizes =
        {
            128,
            256,
            512,
            1024
        };


        std::vector<BenchmarkResult>
            results;


        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "FP32 vs INT8 MatMul Benchmark"
            << std::endl;


        std::cout
            << "============================================"
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
            int matrix_size :
            matrix_sizes
        )
        {
            std::cout
                << "Running "
                << matrix_size
                << " x "
                << matrix_size
                << "..."
                << std::endl;


            results.push_back(
                run_matrix_size(
                    matrix_size
                )
            );
        }


        std::cout
            << "\nPerformance Results"
            << std::endl;


        print_results(
            results
        );


        print_memory_results(
            results
        );


        std::cout
            << "\nInterpretation:"
            << std::endl;


        std::cout
            << "KernelSpeedup = FP32 / INT8-kernel-only"
            << std::endl;


        std::cout
            << "E2ESpeedup = FP32 / (Quantization + INT8 MatMul)"
            << std::endl;


        std::cout
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