#include "cpu/fused_matmul_bias_gelu.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>


// ============================================================
// Error Metrics
// ============================================================

struct ErrorMetrics
{
    double mae =
        0.0;


    double rmse =
        0.0;


    float max_error =
        0.0f;
};


// ============================================================
// Deterministic Initialization
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
// Calculate Error
// ============================================================

ErrorMetrics calculate_error(
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


    double sum_absolute =
        0.0;


    double sum_squared =
        0.0;


    for(
        std::size_t i = 0;
        i < reference.size();
        i++
    )
    {
        const double difference =
            static_cast<double>(
                reference[i]
            )
            -
            static_cast<double>(
                output[i]
            );


        const double absolute_error =
            std::fabs(
                difference
            );


        sum_absolute +=
            absolute_error;


        sum_squared +=
            difference
            *
            difference;


        metrics.max_error =
            std::max(
                metrics.max_error,
                static_cast<float>(
                    absolute_error
                )
            );
    }


    metrics.mae =
        sum_absolute
        /
        static_cast<double>(
            reference.size()
        );


    metrics.rmse =
        std::sqrt(
            sum_squared
            /
            static_cast<double>(
                reference.size()
            )
        );


    return metrics;
}


// ============================================================
// One Test
// ============================================================

ErrorMetrics run_test(
    int M,
    int K,
    int N
)
{
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


    std::vector<float> temp_matmul(
        output_elements
    );


    std::vector<float> temp_bias(
        output_elements
    );


    std::vector<float> reference(
        output_elements
    );


    std::vector<float> fused_output(
        output_elements
    );


    // ========================================================
    // Input Initialization
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
    // Unfused Reference
    // ========================================================

    unfused_matmul_bias_gelu(
        A.data(),
        B.data(),
        bias.data(),
        temp_matmul.data(),
        temp_bias.data(),
        reference.data(),
        M,
        K,
        N
    );


    // ========================================================
    // Fused
    // ========================================================

    fused_matmul_bias_gelu(
        A.data(),
        B.data(),
        bias.data(),
        fused_output.data(),
        M,
        K,
        N
    );


    return
        calculate_error(
            reference,
            fused_output
        );
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        struct TestShape
        {
            int M;
            int K;
            int N;
        };


        const std::vector<TestShape>
            shapes =
        {
            {1,   64,  64},
            {16,  64,  64},
            {32, 128, 128},
            {64, 256, 256},
            {128, 256, 512}
        };


        constexpr double MAE_TOLERANCE =
            1e-6;


        constexpr float MAX_ERROR_TOLERANCE =
            1e-5f;


        bool all_passed =
            true;


        std::cout
            << "\n============================================================"
            << std::endl;


        std::cout
            << "CPU Fused MatMul + Bias + GELU Correctness Test"
            << std::endl;


        std::cout
            << "============================================================"
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
            << "MAE"

            << std::setw(18)
            << "RMSE"

            << std::setw(20)
            << "MaxError"

            << std::setw(12)
            << "Result"

            << std::endl;


        std::cout
            << std::string(
                92,
                '='
            )
            << std::endl;


        for(
            const TestShape& shape :
            shapes
        )
        {
            const ErrorMetrics metrics =
                run_test(
                    shape.M,
                    shape.K,
                    shape.N
                );


            const bool passed =
                metrics.mae
                <=
                MAE_TOLERANCE

                &&

                metrics.max_error
                <=
                MAX_ERROR_TOLERANCE;


            all_passed =
                all_passed
                &&
                passed;


            std::cout
                << std::left

                << std::setw(8)
                << shape.M

                << std::setw(8)
                << shape.K

                << std::setw(8)
                << shape.N

                << std::setw(18)
                << std::scientific
                << std::setprecision(8)
                << metrics.mae

                << std::setw(18)
                << metrics.rmse

                << std::setw(20)
                << metrics.max_error

                << std::setw(12)
                << (
                    passed
                    ?
                    "PASS"
                    :
                    "FAIL"
                )

                << std::endl;
        }


        std::cout
            << "\n============================================================"
            << std::endl;


        if(all_passed)
        {
            std::cout
                << "ALL FUSED OPERATOR TESTS PASSED."
                << std::endl;


            std::cout
                << "============================================================"
                << std::endl;


            return 0;
        }


        std::cerr
            << "FUSED OPERATOR TEST FAILED."
            << std::endl;


        return 1;
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