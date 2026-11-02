#include "quantization/int8_matmul.h"
#include "quantization/quantization.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>


// ============================================================
// FP32 reference MatMul
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
// Deterministic random input
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
// Error metrics
// ============================================================

struct ErrorMetrics
{
    double mae =
        0.0;

    double rmse =
        0.0;

    float max_absolute_error =
        0.0f;

    double mean_relative_error =
        0.0;
};


// ============================================================
// Calculate numerical error
// ============================================================

ErrorMetrics calculate_error_metrics(
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


    ErrorMetrics metrics;


    if(reference.empty())
    {
        return metrics;
    }


    double absolute_error_sum =
        0.0;


    double squared_error_sum =
        0.0;


    double relative_error_sum =
        0.0;


    std::size_t relative_count =
        0;


    for(
        std::size_t i = 0;
        i < reference.size();
        i++
    )
    {
        const double ref =
            static_cast<double>(
                reference[i]
            );


        const double out =
            static_cast<double>(
                output[i]
            );


        const double error =
            std::fabs(
                ref
                -
                out
            );


        absolute_error_sum +=
            error;


        squared_error_sum +=
            error
            *
            error;


        metrics.max_absolute_error =
            std::max(
                metrics.max_absolute_error,
                static_cast<float>(
                    error
                )
            );


        // ----------------------------------------------------
        // Avoid unstable relative error near zero.
        // ----------------------------------------------------

        if(
            std::fabs(ref)
            >
            1e-6
        )
        {
            relative_error_sum +=
                error
                /
                std::fabs(ref);


            relative_count++;
        }
    }


    metrics.mae =
        absolute_error_sum
        /
        static_cast<double>(
            reference.size()
        );


    metrics.rmse =
        std::sqrt(
            squared_error_sum
            /
            static_cast<double>(
                reference.size()
            )
        );


    if(relative_count > 0)
    {
        metrics.mean_relative_error =
            relative_error_sum
            /
            static_cast<double>(
                relative_count
            );
    }


    return metrics;
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        // ====================================================
        // Small matrix first.
        //
        // We want correctness before performance.
        // ====================================================

        constexpr int M =
            64;


        constexpr int K =
            64;


        constexpr int N =
            64;


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


        // ====================================================
        // FP32 matrices
        // ====================================================

        std::vector<float> A(
            A_size
        );


        std::vector<float> B(
            B_size
        );


        std::vector<float> C_reference(
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
        // FP32 reference
        // ====================================================

        fp32_matmul_reference(
            A.data(),
            B.data(),
            C_reference.data(),
            M,
            K,
            N
        );


        // ====================================================
        // Quantize A
        // ====================================================

        const QuantizedTensor A_quantized =
            quantize_fp32_to_int8(
                A.data(),
                A.size()
            );


        // ====================================================
        // Quantize B
        // ====================================================

        const QuantizedTensor B_quantized =
            quantize_fp32_to_int8(
                B.data(),
                B.size()
            );


        // ====================================================
        // INT8 MatMul
        // ====================================================

        int8_matmul(
            A_quantized.data.data(),
            B_quantized.data.data(),
            C_int8.data(),
            M,
            K,
            N,
            A_quantized.params,
            B_quantized.params
        );


        // ====================================================
        // Error
        // ====================================================

        const ErrorMetrics metrics =
            calculate_error_metrics(
                C_reference,
                C_int8
            );


        // ====================================================
        // Print
        // ====================================================

        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "INT8 MatMul Correctness Test"
            << std::endl;


        std::cout
            << "============================================"
            << std::endl;


        std::cout
            << "Matrix Shape:"
            << std::endl;


        std::cout
            << "A = "
            << M
            << " x "
            << K
            << std::endl;


        std::cout
            << "B = "
            << K
            << " x "
            << N
            << std::endl;


        std::cout
            << "C = "
            << M
            << " x "
            << N
            << std::endl;


        std::cout
            << std::endl;


        std::cout
            << "A Scale: "
            << std::fixed
            << std::setprecision(8)
            << A_quantized.params.scale
            << std::endl;


        std::cout
            << "B Scale: "
            << std::fixed
            << std::setprecision(8)
            << B_quantized.params.scale
            << std::endl;


        std::cout
            << std::endl;


        std::cout
            << "MAE: "
            << std::setprecision(8)
            << metrics.mae
            << std::endl;


        std::cout
            << "RMSE: "
            << std::setprecision(8)
            << metrics.rmse
            << std::endl;


        std::cout
            << "Max Absolute Error: "
            << std::setprecision(8)
            << metrics.max_absolute_error
            << std::endl;


        std::cout
            << "Mean Relative Error: "
            << std::setprecision(8)
            << (
                metrics.mean_relative_error
                *
                100.0
            )
            << "%"
            << std::endl;


        // ====================================================
        // Print first 10 outputs
        // ====================================================

        std::cout
            << "\nFirst 10 Outputs:"
            << std::endl;


        std::cout
            << std::left

            << std::setw(18)
            << "FP32"

            << std::setw(18)
            << "INT8 MatMul"

            << std::setw(18)
            << "Abs Error"

            << std::endl;


        std::cout
            << std::string(
                54,
                '-'
            )
            << std::endl;


        const std::size_t print_count =
            std::min<std::size_t>(
                10,
                C_size
            );


        for(
            std::size_t i = 0;
            i < print_count;
            i++
        )
        {
            const float error =
                std::fabs(
                    C_reference[i]
                    -
                    C_int8[i]
                );


            std::cout
                << std::left

                << std::setw(18)
                << std::fixed
                << std::setprecision(6)
                << C_reference[i]

                << std::setw(18)
                << std::fixed
                << std::setprecision(6)
                << C_int8[i]

                << std::setw(18)
                << std::fixed
                << std::setprecision(6)
                << error

                << std::endl;
        }


        // ====================================================
        // Basic pass criterion
        //
        // Quantization is approximate, so this is not tested
        // with an FP32-style 1e-5 tolerance.
        //
        // For random [-1,1] matrices and K=64,
        // MAE should normally remain small.
        // ====================================================

        constexpr double MAX_ALLOWED_MAE =
            0.05;


        if(
            metrics.mae
            >
            MAX_ALLOWED_MAE
        )
        {
            std::cerr
                << "\nINT8 MatMul test FAILED."
                << std::endl;


            std::cerr
                << "MAE is larger than allowed threshold."
                << std::endl;


            return 1;
        }


        std::cout
            << "\nINT8 MatMul test PASSED."
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