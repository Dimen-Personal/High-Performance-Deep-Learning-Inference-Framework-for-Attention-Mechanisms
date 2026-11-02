#include "quantization/int8_matmul.h"
#include "quantization/quantization.h"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
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
// Result
// ============================================================

struct ScalingResult
{
    int matrix_size =
        0;

    int threads =
        0;

    double time_ms =
        0.0;

    double speedup =
        0.0;

    double efficiency =
        0.0;

    double mae =
        0.0;

    bool correct =
        false;
};


// ============================================================
// Deterministic initialization
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
// FP32 reference
// ============================================================

void fp32_matmul_reference(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
)
{
    for(
        int i = 0;
        i < M;
        i++
    )
    {
        for(
            int j = 0;
            j < N;
            j++
        )
        {
            float sum =
                0.0f;


            for(
                int k = 0;
                k < K;
                k++
            )
            {
                sum +=
                    A[
                        i * K
                        + k
                    ]
                    *
                    B[
                        k * N
                        + j
                    ];
            }


            C[
                i * N
                + j
            ] =
                sum;
        }
    }
}


// ============================================================
// MAE
// ============================================================

double calculate_mae(
    const std::vector<float>& reference,
    const std::vector<float>& output
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


    double total =
        0.0;


    for(
        std::size_t i = 0;
        i < reference.size();
        i++
    )
    {
        total +=
            std::fabs(
                static_cast<double>(
                    reference[i]
                )
                -
                static_cast<double>(
                    output[i]
                )
            );
    }


    return
        total
        /
        static_cast<double>(
            reference.size()
        );
}


// ============================================================
// Print OpenMP team
// ============================================================

void print_thread_information(
    int thread_count
)
{
    std::cout
        << "OpenMP team for "
        << thread_count
        << " requested threads:"
        << std::endl;


    #pragma omp parallel \
        num_threads(thread_count)
    {
        const int tid =
            omp_get_thread_num();


        const int total =
            omp_get_num_threads();


        #pragma omp critical
        {
            std::cout
                << "  Thread "
                << tid
                << " / "
                << total
                << std::endl;
        }
    }


    std::cout
        << std::endl;
}


// ============================================================
// Benchmark one thread count
// ============================================================

double benchmark_int8_threads(
    const QuantizedTensor& A,
    const QuantizedTensor& B,
    float* C,
    int M,
    int K,
    int N,
    int thread_count
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
        int8_matmul_openmp(
            A.data.data(),
            B.data.data(),
            C,
            M,
            K,
            N,
            A.params,
            B.params,
            thread_count
        );
    }


    // --------------------------------------------------------
    // Benchmark
    // --------------------------------------------------------

    const double start =
        omp_get_wtime();


    for(
        int run = 0;
        run < BENCHMARK_RUNS;
        run++
    )
    {
        int8_matmul_openmp(
            A.data.data(),
            B.data.data(),
            C,
            M,
            K,
            N,
            A.params,
            B.params,
            thread_count
        );
    }


    const double stop =
        omp_get_wtime();


    return
        (
            stop
            -
            start
        )
        *
        1000.0
        /
        static_cast<double>(
            BENCHMARK_RUNS
        );
}


// ============================================================
// Run one matrix size
// ============================================================

std::vector<ScalingResult>
run_matrix_size(
    int matrix_size,
    const std::vector<int>& thread_counts
)
{
    const int M =
        matrix_size;


    const int K =
        matrix_size;


    const int N =
        matrix_size;


    const std::size_t matrix_elements =
        static_cast<std::size_t>(
            matrix_size
        )
        *
        matrix_size;


    std::vector<float> A_fp32(
        matrix_elements
    );


    std::vector<float> B_fp32(
        matrix_elements
    );


    std::vector<float> C_reference(
        matrix_elements
    );


    std::vector<float> C_int8(
        matrix_elements
    );


    // ========================================================
    // Initialize
    // ========================================================

    initialize_matrix(
        A_fp32,
        1
    );


    initialize_matrix(
        B_fp32,
        2
    );


    // ========================================================
    // FP32 reference
    // ========================================================

    std::cout
        << "Generating FP32 reference..."
        << std::endl;


    fp32_matmul_reference(
        A_fp32.data(),
        B_fp32.data(),
        C_reference.data(),
        M,
        K,
        N
    );


    // ========================================================
    // Quantize once
    //
    // Scaling benchmark should measure INT8 compute scaling,
    // not repeated quantization cost.
    // ========================================================

    const QuantizedTensor A_int8 =
        quantize_fp32_to_int8(
            A_fp32.data(),
            A_fp32.size()
        );


    const QuantizedTensor B_int8 =
        quantize_fp32_to_int8(
            B_fp32.data(),
            B_fp32.size()
        );


    std::vector<ScalingResult>
        results;


    double single_thread_time =
        0.0;


    // ========================================================
    // Benchmark each worker count
    // ========================================================

    for(
        int thread_count :
        thread_counts
    )
    {
        print_thread_information(
            thread_count
        );


        const double time_ms =
            benchmark_int8_threads(
                A_int8,
                B_int8,
                C_int8.data(),
                M,
                K,
                N,
                thread_count
            );


        const double mae =
            calculate_mae(
                C_reference,
                C_int8
            );


        ScalingResult result;


        result.matrix_size =
            matrix_size;


        result.threads =
            thread_count;


        result.time_ms =
            time_ms;


        result.mae =
            mae;


        // ----------------------------------------------------
        // Quantized result is approximate.
        //
        // Current baseline MAE is expected to remain small.
        // ----------------------------------------------------

        result.correct =
            mae
            <
            0.1;


        results.push_back(
            result
        );


        if(thread_count == 1)
        {
            single_thread_time =
                time_ms;
        }
    }


    if(single_thread_time <= 0.0)
    {
        throw std::runtime_error(
            "Single-thread baseline missing."
        );
    }


    // ========================================================
    // Speedup / efficiency
    // ========================================================

    for(
        ScalingResult& result :
        results
    )
    {
        result.speedup =
            single_thread_time
            /
            result.time_ms;


        result.efficiency =
            result.speedup
            /
            static_cast<double>(
                result.threads
            );
    }


    return results;
}


// ============================================================
// Print results
// ============================================================

void print_results(
    const std::vector<ScalingResult>& results
)
{
    std::cout
        << std::left

        << std::setw(10)
        << "Size"

        << std::setw(12)
        << "Threads"

        << std::setw(16)
        << "Time(ms)"

        << std::setw(14)
        << "Speedup"

        << std::setw(16)
        << "Efficiency"

        << std::setw(14)
        << "MAE"

        << std::setw(12)
        << "Correct"

        << std::endl;


    std::cout
        << std::string(
            94,
            '-'
        )
        << std::endl;


    for(
        const ScalingResult& result :
        results
    )
    {
        std::cout
            << std::left

            << std::setw(10)
            << result.matrix_size

            << std::setw(12)
            << result.threads

            << std::setw(16)
            << std::fixed
            << std::setprecision(4)
            << result.time_ms

            << std::setw(14)
            << std::fixed
            << std::setprecision(4)
            << result.speedup

            << std::setw(16)
            << std::fixed
            << std::setprecision(2)
            << (
                result.efficiency
                *
                100.0
            )

            << std::setw(14)
            << std::fixed
            << std::setprecision(6)
            << result.mae

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
// Main
// ============================================================

int main()
{
    try
    {
        omp_set_dynamic(
            0
        );


        const std::vector<int>
            thread_counts =
        {
            1,
            2,
            4,
            8
        };


        // ====================================================
        // We already know INT8 becomes useful at larger sizes.
        //
        // Therefore start with:
        //
        // 256
        // 512
        // 1024
        // ====================================================

        const std::vector<int>
            matrix_sizes =
        {
            256,
            512,
            1024
        };


        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "INT8 MatMul OpenMP Scaling Benchmark"
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
            << "OpenMP Max Threads: "
            << omp_get_max_threads()
            << std::endl;


        std::cout
            << "OpenMP Dynamic Threads: "
            << omp_get_dynamic()
            << std::endl;


        std::cout
            << std::endl;


        for(
            int matrix_size :
            matrix_sizes
        )
        {
            std::cout
                << "\n============================================"
                << std::endl;


            std::cout
                << "Matrix Size: "
                << matrix_size
                << " x "
                << matrix_size
                << std::endl;


            std::cout
                << "============================================"
                << std::endl;


            const std::vector<ScalingResult>
                results =
                    run_matrix_size(
                        matrix_size,
                        thread_counts
                    );


            std::cout
                << "\nResults"
                << std::endl;


            print_results(
                results
            );
        }


        std::cout
            << "\nInterpretation:"
            << std::endl;


        std::cout
            << "Speedup = T1 / Tp"
            << std::endl;


        std::cout
            << "Efficiency = Speedup / Threads"
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