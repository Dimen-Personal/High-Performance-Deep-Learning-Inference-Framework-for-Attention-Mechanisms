#include "cpu/fused_matmul_bias_gelu.h"
#include "cpu/cache_aware_attention.h"
#include "ops/operator.h"

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
// Error Calculation
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
            "Reference/output size mismatch."
        );
    }


    ErrorMetrics metrics;


    if(reference.empty())
    {
        return metrics;
    }


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
// Stage A
// MatMul + Bias + GELU Fusion Correctness
// ============================================================

ErrorMetrics run_fusion_test(
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
// Stage B
// Cache-Aware Attention Correctness
// ============================================================

ErrorMetrics run_cache_aware_attention_test(
    int seq_len,
    int head_dim
)
{
    const std::size_t elements =
        static_cast<std::size_t>(
            seq_len
        )
        *
        head_dim;


    std::vector<float> Q(
        elements
    );


    std::vector<float> K(
        elements
    );


    std::vector<float> V(
        elements
    );


    std::vector<float> reference(
        elements
    );


    std::vector<float> tiled_output(
        elements
    );


    initialize_input(
        Q,
        1
    );


    initialize_input(
        K,
        17
    );


    initialize_input(
        V,
        37
    );


    // ========================================================
    // Original CPU Attention Baseline
    //
    // QK^T
    //   ↓
    // scores[N,N]
    //   ↓
    // Softmax
    //   ↓
    // weights[N,N]
    //   ↓
    // weights × V
    // ========================================================

    scaled_dot_product_attention(
        Q.data(),
        K.data(),
        V.data(),
        reference.data(),
        seq_len,
        head_dim
    );


    // ========================================================
    // Cache-Aware Tiled Attention
    //
    // Query block = 16
    // Key block   = 32
    // ========================================================

    cache_aware_attention_tiled(
        Q.data(),
        K.data(),
        V.data(),
        tiled_output.data(),
        seq_len,
        head_dim,
        16,
        32
    );


    return
        calculate_error(
            reference,
            tiled_output
        );
}


// ============================================================
// Print Stage A Header
// ============================================================

void print_stage_a_header()
{
    std::cout
        << "\n============================================================"
        << std::endl;


    std::cout
        << "Stage A - CPU MatMul + Bias + GELU Fusion Correctness"
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
}


// ============================================================
// Print Stage B Header
// ============================================================

void print_stage_b_header()
{
    std::cout
        << "\n============================================================"
        << std::endl;


    std::cout
        << "Stage B - CPU Cache-Aware Tiled Attention Correctness"
        << std::endl;


    std::cout
        << "============================================================"
        << std::endl;


    std::cout
        << "Query Block Size: 16"
        << std::endl;


    std::cout
        << "Key Block Size: 32"
        << std::endl;


    std::cout
        << "Head Dimension: 64"
        << std::endl;


    std::cout
        << std::endl;


    std::cout
        << std::left

        << std::setw(12)
        << "SeqLen"

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
            80,
            '='
        )
        << std::endl;
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        bool all_passed =
            true;


        // ====================================================
        // Stage A
        // ====================================================

        struct FusionShape
        {
            int M;
            int K;
            int N;
        };


        const std::vector<FusionShape>
            fusion_shapes =
        {
            {1,   64,  64},
            {16,  64,  64},
            {32, 128, 128},
            {64, 256, 256},
            {128, 256, 512}
        };


        constexpr double FUSION_MAE_TOLERANCE =
            1e-6;


        constexpr float FUSION_MAX_ERROR_TOLERANCE =
            1e-5f;


        print_stage_a_header();


        for(
            const FusionShape& shape :
            fusion_shapes
        )
        {
            const ErrorMetrics metrics =
                run_fusion_test(
                    shape.M,
                    shape.K,
                    shape.N
                );


            const bool passed =
                metrics.mae
                <=
                FUSION_MAE_TOLERANCE

                &&

                metrics.max_error
                <=
                FUSION_MAX_ERROR_TOLERANCE;


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


        // ====================================================
        // Stage B
        // ====================================================

        constexpr int ATTENTION_HEAD_DIM =
            64;


        const std::vector<int>
            sequence_lengths =
        {
            1,
            2,

            15,
            16,
            17,

            31,
            32,
            33,

            63,
            64,
            65,

            128,
            256,
            512
        };


        constexpr double ATTENTION_MAE_TOLERANCE =
            1e-4;


        constexpr float ATTENTION_MAX_ERROR_TOLERANCE =
            1e-3f;


        print_stage_b_header();


        for(
            int seq_len :
            sequence_lengths
        )
        {
            const ErrorMetrics metrics =
                run_cache_aware_attention_test(
                    seq_len,
                    ATTENTION_HEAD_DIM
                );


            const bool passed =
                metrics.mae
                <=
                ATTENTION_MAE_TOLERANCE

                &&

                metrics.max_error
                <=
                ATTENTION_MAX_ERROR_TOLERANCE;


            all_passed =
                all_passed
                &&
                passed;


            std::cout
                << std::left

                << std::setw(12)
                << seq_len

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


        // ====================================================
        // Final Result
        // ====================================================

        std::cout
            << "\n============================================================"
            << std::endl;


        if(all_passed)
        {
            std::cout
                << "ALL PHASE 5 ALTERNATIVE CORRECTNESS TESTS PASSED."
                << std::endl;


            std::cout
                << "============================================================"
                << std::endl;


            return 0;
        }


        std::cerr
            << "PHASE 5 ALTERNATIVE CORRECTNESS TEST FAILED."
            << std::endl;


        std::cout
            << "============================================================"
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