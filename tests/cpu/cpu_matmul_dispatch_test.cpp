#include "cpu/cpu_matmul_dispatch.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>


// ============================================================
// Reference FP32 MatMul
// ============================================================

void reference_matmul(
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
// Calculate MAE
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


    double total_error =
        0.0;


    for(
        std::size_t i = 0;
        i < reference.size();
        i++
    )
    {
        total_error +=
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
        total_error
        /
        static_cast<double>(
            reference.size()
        );
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        constexpr int M =
            64;


        constexpr int K =
            64;


        constexpr int N =
            64;


        constexpr int THREAD_COUNT =
            4;


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


        std::vector<float> A(
            A_size
        );


        std::vector<float> B(
            B_size
        );


        std::vector<float> C_reference(
            C_size
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


        // ====================================================
        // Reference
        // ====================================================

        reference_matmul(
            A.data(),
            B.data(),
            C_reference.data(),
            M,
            K,
            N
        );


        // ====================================================
        // FP32 dispatch
        // ====================================================

        CpuMatmulConfig fp32_config;


        fp32_config.mode =
            CpuMatmulMode::FP32;


        fp32_config.thread_count =
            THREAD_COUNT;


        cpu_matmul_dispatch(
            A.data(),
            B.data(),
            C_fp32.data(),
            M,
            K,
            N,
            fp32_config
        );


        // ====================================================
        // INT8 dispatch
        // ====================================================

        CpuMatmulConfig int8_config;


        int8_config.mode =
            CpuMatmulMode::INT8;


        int8_config.thread_count =
            THREAD_COUNT;


        cpu_matmul_dispatch(
            A.data(),
            B.data(),
            C_int8.data(),
            M,
            K,
            N,
            int8_config
        );


        // ====================================================
        // Error
        // ====================================================

        const double fp32_mae =
            calculate_mae(
                C_reference,
                C_fp32
            );


        const double int8_mae =
            calculate_mae(
                C_reference,
                C_int8
            );


        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "CPU MatMul Dispatch Test"
            << std::endl;


        std::cout
            << "============================================"
            << std::endl;


        std::cout
            << "Shape: "
            << M
            << " x "
            << K
            << " x "
            << N
            << std::endl;


        std::cout
            << "Threads: "
            << THREAD_COUNT
            << std::endl;


        std::cout
            << std::endl;


        std::cout
            << std::fixed
            << std::setprecision(8);


        std::cout
            << "FP32 Dispatch MAE: "
            << fp32_mae
            << std::endl;


        std::cout
            << "INT8 Dispatch MAE: "
            << int8_mae
            << std::endl;


        // ====================================================
        // FP32 correctness
        // ====================================================

        if(
            fp32_mae
            >
            1e-5
        )
        {
            std::cerr
                << "FP32 dispatch FAILED."
                << std::endl;


            return 1;
        }


        // ====================================================
        // INT8 correctness
        // ====================================================

        if(
            int8_mae
            >
            0.05
        )
        {
            std::cerr
                << "INT8 dispatch FAILED."
                << std::endl;


            return 1;
        }


        std::cout
            << "\nFP32 dispatch: PASS"
            << std::endl;


        std::cout
            << "INT8 dispatch: PASS"
            << std::endl;


        std::cout
            << "\nCPU MatMul unified dispatch PASSED."
            << std::endl;


        return 0;
    }
    catch(
        const std::exception& exception
    )
    {
        std::cerr
            << "Test exception: "
            << exception.what()
            << std::endl;


        return 1;
    }
}