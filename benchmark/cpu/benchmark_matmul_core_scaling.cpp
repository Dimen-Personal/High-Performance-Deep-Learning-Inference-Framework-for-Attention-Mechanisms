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

constexpr int WARMUP_RUNS = 3;

constexpr int BENCHMARK_RUNS = 10;


// ============================================================
// Benchmark result
// ============================================================

struct BenchmarkResult
{
    int matrix_size = 0;

    int threads = 0;

    double average_ms = 0.0;

    double speedup = 0.0;

    double efficiency = 0.0;

    bool correct = false;
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


    for(float& value : matrix)
    {
        value =
            distribution(
                generator
            );
    }
}


// ============================================================
// Single-thread reference MatMul
//
// A: [M, K]
// B: [K, N]
// C: [M, N]
// ============================================================

void matmul_reference(
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
// OpenMP MatMul
//
// Each thread receives a different subset of output rows.
//
// schedule(static):
// - workload per row is approximately equal
// - deterministic work distribution
// - low scheduling overhead
// ============================================================

void matmul_openmp(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
    int thread_count
)
{
    #pragma omp parallel for \
        num_threads(thread_count) \
        schedule(static)

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
// Verify output
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


    float max_difference =
        0.0f;


    for(
        std::size_t i = 0;
        i < reference.size();
        i++
    )
    {
        const float difference =
            std::fabs(
                reference[i]
                -
                output[i]
            );


        max_difference =
            std::max(
                max_difference,
                difference
            );


        if(
            difference
            >
            tolerance
        )
        {
            std::cerr
                << "Mismatch at index "
                << i
                << std::endl;


            std::cerr
                << "Reference: "
                << reference[i]
                << std::endl;


            std::cerr
                << "Output: "
                << output[i]
                << std::endl;


            std::cerr
                << "Difference: "
                << difference
                << std::endl;


            return false;
        }
    }


    return true;
}


// ============================================================
// Print OpenMP thread information
//
// This verifies that OpenMP actually created the requested
// number of worker threads.
//
// macOS does not expose a Linux-style stable sched_getcpu()
// interface for exact core-ID verification, so here we verify
// the OpenMP worker count instead.
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
        const int thread_id =
            omp_get_thread_num();


        const int total_threads =
            omp_get_num_threads();


        #pragma omp critical
        {
            std::cout
                << "  Thread "
                << thread_id
                << " / "
                << total_threads
                << std::endl;
        }
    }


    std::cout
        << std::endl;
}


// ============================================================
// Benchmark one thread count
// ============================================================

double benchmark_thread_count(
    const float* A,
    const float* B,
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
        matmul_openmp(
            A,
            B,
            C,
            M,
            K,
            N,
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
        matmul_openmp(
            A,
            B,
            C,
            M,
            K,
            N,
            thread_count
        );
    }


    const double end =
        omp_get_wtime();


    const double total_seconds =
        end
        -
        start;


    const double average_seconds =
        total_seconds
        /
        static_cast<double>(
            BENCHMARK_RUNS
        );


    return
        average_seconds
        *
        1000.0;
}


// ============================================================
// Run one matrix size
// ============================================================

std::vector<BenchmarkResult>
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
        static_cast<std::size_t>(
            matrix_size
        );


    std::vector<float> A(
        matrix_elements
    );


    std::vector<float> B(
        matrix_elements
    );


    std::vector<float> reference(
        matrix_elements
    );


    std::vector<float> output(
        matrix_elements
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
    // Generate reference output
    // ========================================================

    std::cout
        << "Generating reference output..."
        << std::endl;


    matmul_reference(
        A.data(),
        B.data(),
        reference.data(),
        M,
        K,
        N
    );


    // ========================================================
    // Benchmark
    // ========================================================

    std::vector<BenchmarkResult>
        results;


    double single_thread_time =
        0.0;


    for(
        int thread_count :
        thread_counts
    )
    {
        print_thread_information(
            thread_count
        );


        const double average_ms =
            benchmark_thread_count(
                A.data(),
                B.data(),
                output.data(),
                M,
                K,
                N,
                thread_count
            );


        const bool correct =
            compare_outputs(
                reference,
                output,
                1e-3f
            );


        BenchmarkResult result;


        result.matrix_size =
            matrix_size;


        result.threads =
            thread_count;


        result.average_ms =
            average_ms;


        result.correct =
            correct;


        results.push_back(
            result
        );


        if(
            thread_count
            ==
            1
        )
        {
            single_thread_time =
                average_ms;
        }
    }


    if(
        single_thread_time
        <=
        0.0
    )
    {
        throw std::runtime_error(
            "Single-thread baseline was not measured."
        );
    }


    // ========================================================
    // Calculate speedup and efficiency
    // ========================================================

    for(
        BenchmarkResult& result :
        results
    )
    {
        result.speedup =
            single_thread_time
            /
            result.average_ms;


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
// Print benchmark table
// ============================================================

void print_results(
    const std::vector<BenchmarkResult>& results
)
{
    std::cout
        << std::left

        << std::setw(12)
        << "Matrix"

        << std::setw(12)
        << "Threads"

        << std::setw(16)
        << "Time(ms)"

        << std::setw(14)
        << "Speedup"

        << std::setw(16)
        << "Efficiency"

        << std::setw(12)
        << "Correct"

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

            << std::setw(12)
            << result.matrix_size

            << std::setw(12)
            << result.threads

            << std::setw(16)
            << std::fixed
            << std::setprecision(4)
            << result.average_ms

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
        // ----------------------------------------------------
        // Important:
        //
        // Disable OpenMP dynamic thread adjustment.
        //
        // If enabled, OpenMP runtime may decide to use fewer
        // threads than requested.
        // ----------------------------------------------------

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


        // ----------------------------------------------------
        // Start with 512.
        //
        // This size is large enough for meaningful CPU timing
        // while still being reasonable on a laptop.
        //
        // We can later test 256 / 512 / 1024 separately.
        // ----------------------------------------------------

        constexpr int matrix_size =
            512;


        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "CPU MatMul OpenMP Scaling Benchmark"
            << std::endl;


        std::cout
            << "============================================"
            << std::endl;


        std::cout
            << "Matrix Size: "
            << matrix_size
            << " x "
            << matrix_size
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


        const std::vector<BenchmarkResult>
            results =
                run_matrix_size(
                    matrix_size,
                    thread_counts
                );


        std::cout
            << "\nFinal Results"
            << std::endl;


        std::cout
            << "============================================"
            << std::endl;


        print_results(
            results
        );


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