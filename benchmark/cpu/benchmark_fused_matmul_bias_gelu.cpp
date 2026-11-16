#include "cpu/fused_matmul_bias_gelu.h"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>


// ============================================================
// Benchmark Configuration
// ============================================================

constexpr int WARMUP_RUNS =
    3;


constexpr int BENCHMARK_RUNS =
    10;


// ============================================================
// Benchmark Result
// ============================================================

struct BenchmarkResult
{
    int M =
        0;

    int K =
        0;

    int N =
        0;


    double unfused_ms =
        0.0;


    double fused_ms =
        0.0;


    double speedup =
        0.0;


    double unfused_temporary_memory_mb =
        0.0;


    double fused_temporary_memory_mb =
        0.0;


    double unfused_intermediate_traffic_mb =
        0.0;


    double fused_intermediate_traffic_mb =
        0.0;
};


// ============================================================
// Deterministic Input Initialization
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
// Generic CPU Benchmark
// ============================================================

template<typename Function>
double benchmark_function(
    Function function
)
{
    // ========================================================
    // Warmup
    // ========================================================

    for(
        int i = 0;
        i < WARMUP_RUNS;
        i++
    )
    {
        function();
    }


    // ========================================================
    // Timed Runs
    // ========================================================

    const auto start =
        std::chrono::high_resolution_clock::now();


    for(
        int i = 0;
        i < BENCHMARK_RUNS;
        i++
    )
    {
        function();
    }


    const auto end =
        std::chrono::high_resolution_clock::now();


    const std::chrono::duration<double, std::milli>
        elapsed =
            end
            -
            start;


    return
        elapsed.count()
        /
        static_cast<double>(
            BENCHMARK_RUNS
        );
}


// ============================================================
// Run One Shape
// ============================================================

BenchmarkResult run_shape(
    int M,
    int K,
    int N
)
{
    // ========================================================
    // Tensor Sizes
    // ========================================================

    const std::size_t a_elements =
        static_cast<std::size_t>(
            M
        )
        *
        K;


    const std::size_t b_elements =
        static_cast<std::size_t>(
            K
        )
        *
        N;


    const std::size_t output_elements =
        static_cast<std::size_t>(
            M
        )
        *
        N;


    // ========================================================
    // Buffers
    // ========================================================

    std::vector<float> A(
        a_elements
    );


    std::vector<float> B(
        b_elements
    );


    std::vector<float> bias(
        N
    );


    // --------------------------------------------------------
    // Unfused temporary buffers
    // --------------------------------------------------------

    std::vector<float> temp_matmul(
        output_elements
    );


    std::vector<float> temp_bias(
        output_elements
    );


    std::vector<float> unfused_output(
        output_elements
    );


    // --------------------------------------------------------
    // Fused output
    // --------------------------------------------------------

    std::vector<float> fused_output(
        output_elements
    );


    // ========================================================
    // Initialize
    // ========================================================

    initialize_input(
        A,
        1
    );


    initialize_input(
        B,
        17
    );


    initialize_input(
        bias,
        31
    );


    // ========================================================
    // Unfused Benchmark
    // ========================================================

    const double unfused_ms =
        benchmark_function(
            [&]()
            {
                unfused_matmul_bias_gelu(
                    A.data(),
                    B.data(),
                    bias.data(),
                    temp_matmul.data(),
                    temp_bias.data(),
                    unfused_output.data(),
                    M,
                    K,
                    N
                );
            }
        );


    // ========================================================
    // Fused Benchmark
    // ========================================================

    const double fused_ms =
        benchmark_function(
            [&]()
            {
                fused_matmul_bias_gelu(
                    A.data(),
                    B.data(),
                    bias.data(),
                    fused_output.data(),
                    M,
                    K,
                    N
                );
            }
        );


    // ========================================================
    // Memory Analysis
    // ========================================================

    const double output_bytes =
        static_cast<double>(
            output_elements
        )
        *
        sizeof(float);


    // ========================================================
    // Unfused Temporary Memory
    //
    // temp_matmul[M,N]
    //
    // +
    //
    // temp_bias[M,N]
    // ========================================================

    const double unfused_temporary_bytes =
        2.0
        *
        output_bytes;


    // ========================================================
    // Fused Temporary Memory
    //
    // No M×N intermediate tensors.
    // ========================================================

    const double fused_temporary_bytes =
        0.0;


    // ========================================================
    // Estimated Intermediate Traffic
    //
    // Unfused:
    //
    // 1. MatMul writes temp_matmul
    //
    // 2. Bias reads temp_matmul
    //
    // 3. Bias writes temp_bias
    //
    // 4. GELU reads temp_bias
    //
    // 5. GELU writes final output
    //
    // = 5 × M × N floats
    //
    //
    // Fused:
    //
    // only final output write
    //
    // = 1 × M × N floats
    //
    // NOTE:
    //
    // This is only intermediate/output traffic.
    //
    // It intentionally excludes A/B reads because both
    // implementations perform the same MatMul.
    // ========================================================

    const double unfused_traffic_bytes =
        5.0
        *
        output_bytes;


    const double fused_traffic_bytes =
        1.0
        *
        output_bytes;


    // ========================================================
    // Build Result
    // ========================================================

    BenchmarkResult result;


    result.M =
        M;


    result.K =
        K;


    result.N =
        N;


    result.unfused_ms =
        unfused_ms;


    result.fused_ms =
        fused_ms;


    result.speedup =
        unfused_ms
        /
        fused_ms;


    result.unfused_temporary_memory_mb =
        unfused_temporary_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.fused_temporary_memory_mb =
        fused_temporary_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.unfused_intermediate_traffic_mb =
        unfused_traffic_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.fused_intermediate_traffic_mb =
        fused_traffic_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    return result;
}


// ============================================================
// Print Performance Table
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

        << std::setw(8)
        << "M"

        << std::setw(8)
        << "K"

        << std::setw(8)
        << "N"

        << std::setw(18)
        << "Unfused(ms)"

        << std::setw(18)
        << "Fused(ms)"

        << std::setw(16)
        << "Speedup"

        << std::endl;


    std::cout
        << std::string(
            76,
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

            << std::setw(8)
            << result.M

            << std::setw(8)
            << result.K

            << std::setw(8)
            << result.N

            << std::setw(18)
            << std::fixed
            << std::setprecision(6)
            << result.unfused_ms

            << std::setw(18)
            << std::fixed
            << std::setprecision(6)
            << result.fused_ms

            << std::setw(16)
            << std::fixed
            << std::setprecision(4)
            << result.speedup

            << std::endl;
    }
}


// ============================================================
// Print Memory Table
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

        << std::setw(8)
        << "M"

        << std::setw(8)
        << "K"

        << std::setw(8)
        << "N"

        << std::setw(22)
        << "UnfusedTemp(MB)"

        << std::setw(20)
        << "FusedTemp(MB)"

        << std::setw(24)
        << "UnfusedTraffic(MB)"

        << std::setw(22)
        << "FusedTraffic(MB)"

        << std::endl;


    std::cout
        << std::string(
            112,
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

            << std::setw(8)
            << result.M

            << std::setw(8)
            << result.K

            << std::setw(8)
            << result.N

            << std::setw(22)
            << std::fixed
            << std::setprecision(4)
            << result.unfused_temporary_memory_mb

            << std::setw(20)
            << std::fixed
            << std::setprecision(4)
            << result.fused_temporary_memory_mb

            << std::setw(24)
            << std::fixed
            << std::setprecision(4)
            << result.unfused_intermediate_traffic_mb

            << std::setw(22)
            << std::fixed
            << std::setprecision(4)
            << result.fused_intermediate_traffic_mb

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
        struct BenchmarkShape
        {
            int M;
            int K;
            int N;
        };


        // ====================================================
        // Shapes
        //
        // These progressively increase the output matrix size
        // and therefore increase intermediate memory traffic.
        // ====================================================

        const std::vector<BenchmarkShape>
            shapes =
        {
            {32,  128, 128},
            {64,  256, 256},
            {128, 256, 512},
            {256, 512, 512},
            {512, 512, 512},

            {768, 512, 768},
            {1024, 512, 1024}
        };


        std::vector<BenchmarkResult>
            results;


        std::cout
            << "\n============================================================"
            << std::endl;


        std::cout
            << "CPU MatMul + Bias + GELU Fusion Benchmark"
            << std::endl;


        std::cout
            << "============================================================"
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
            << "\nCompared Implementations:"
            << std::endl;


        std::cout
            << "1. Unfused: MatMul -> Bias -> GELU"
            << std::endl;


        std::cout
            << "2. Fused: MatMul + Bias + GELU"
            << std::endl;


        std::cout
            << std::endl;


        // ====================================================
        // Run
        // ====================================================

        for(
            const BenchmarkShape& shape :
            shapes
        )
        {
            std::cout
                << "Running M="
                << shape.M
                << ", K="
                << shape.K
                << ", N="
                << shape.N
                << "..."
                << std::endl;


            results.push_back(
                run_shape(
                    shape.M,
                    shape.K,
                    shape.N
                )
            );
        }


        // ====================================================
        // Results
        // ====================================================

        print_performance_results(
            results
        );


        print_memory_results(
            results
        );


        // ====================================================
        // Notes
        // ====================================================

        std::cout
            << "\nNotes:"
            << std::endl;


        std::cout
            << "Speedup = UnfusedTime / FusedTime."
            << std::endl;


        std::cout
            << "Speedup > 1.0 means fusion is faster."
            << std::endl;


        std::cout
            << "Temporary memory counts only MxN intermediate tensors."
            << std::endl;


        std::cout
            << "Traffic values are logical intermediate/output traffic estimates."
            << std::endl;


        std::cout
            << "A/B MatMul reads are excluded because both versions require them."
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