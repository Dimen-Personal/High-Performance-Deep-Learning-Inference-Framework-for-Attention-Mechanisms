// ============================================================
// Flash Attention Correctness Regression Test
//
// Reference:
//
//     AttentionScores
//          ↓
//     AttentionSoftmax
//          ↓
//     MatMul(weights, V)
//
// Compared implementations:
//
//     1. Original Tiled Flash Attention
//
//     2. Warp-Optimized Tiled Flash Attention
//
//     3. Multi-Query Optimized Flash Attention
//
// Every implementation is independently compared with the
// SAME baseline output.
// ============================================================

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
// CUDA Error Helper
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
                +                                                            \
                cudaGetErrorString(error)                                    \
            );                                                               \
        }                                                                    \
    } while(false)


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
// Result for all Flash Attention versions
// ============================================================

struct FlashAttentionTestResult
{
    ErrorMetrics original_tiled;

    ErrorMetrics warp_optimized;

    ErrorMetrics multiquery_optimized;
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
// Run One Sequence Length
// ============================================================

FlashAttentionTestResult run_test(
    int seq_len,
    int head_dim
)
{
    // ========================================================
    // Tensor Sizes
    // ========================================================

    const std::size_t qkv_elements =
        static_cast<std::size_t>(
            seq_len
        )
        *
        static_cast<std::size_t>(
            head_dim
        );


    const std::size_t attention_elements =
        static_cast<std::size_t>(
            seq_len
        )
        *
        static_cast<std::size_t>(
            seq_len
        );


    // ========================================================
    // Host Buffers
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


    std::vector<float> h_reference(
        qkv_elements
    );


    std::vector<float> h_original_tiled(
        qkv_elements
    );


    std::vector<float> h_warp_optimized(
        qkv_elements
    );


    std::vector<float> h_multiquery_optimized(
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
    // Device Buffers
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


    float* d_reference =
        nullptr;


    float* d_original_tiled =
        nullptr;


    float* d_warp_optimized =
        nullptr;


    float* d_multiquery_optimized =
        nullptr;


    // ========================================================
    // Allocate Q / K / V
    // ========================================================

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


    // ========================================================
    // Baseline Intermediate Buffers
    // ========================================================

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


    // ========================================================
    // Output Buffers
    // ========================================================

    CUDA_CHECK(
        cudaMalloc(
            &d_reference,
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_original_tiled,
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_warp_optimized,
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            &d_multiquery_optimized,
            qkv_elements
            *
            sizeof(float)
        )
    );


    // ========================================================
    // H2D
    // ========================================================

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
    // Baseline
    //
    // QK^T
    //   ↓
    // Softmax
    //   ↓
    // weights × V
    // ========================================================

    cuda_attention_scores(
        d_Q,
        d_K,
        d_scores,
        seq_len,
        head_dim
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    cuda_attention_softmax(
        d_scores,
        d_weights,
        seq_len
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    cuda_matmul(
        d_weights,
        d_V,
        d_reference,
        seq_len,
        seq_len,
        head_dim
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // 1. Original Tiled Flash Attention
    // ========================================================

    cuda_flash_attention_tiled(
        d_Q,
        d_K,
        d_V,
        d_original_tiled,
        seq_len,
        head_dim
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // 2. Warp-Optimized Tiled Flash Attention
    // ========================================================

    cuda_flash_attention_tiled_optimized(
        d_Q,
        d_K,
        d_V,
        d_warp_optimized,
        seq_len,
        head_dim
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // 3. Multi-Query Optimized Flash Attention
    // ========================================================

    cuda_flash_attention_multiquery_optimized(
        d_Q,
        d_K,
        d_V,
        d_multiquery_optimized,
        seq_len,
        head_dim
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


    // ========================================================
    // Synchronize
    // ========================================================

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ========================================================
    // D2H
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_reference.data(),
            d_reference,
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_original_tiled.data(),
            d_original_tiled,
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_warp_optimized.data(),
            d_warp_optimized,
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_multiquery_optimized.data(),
            d_multiquery_optimized,
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // Calculate Error for Every Version
    // ========================================================

    FlashAttentionTestResult result;


    result.original_tiled =
        calculate_error(
            h_reference,
            h_original_tiled
        );


    result.warp_optimized =
        calculate_error(
            h_reference,
            h_warp_optimized
        );


    result.multiquery_optimized =
        calculate_error(
            h_reference,
            h_multiquery_optimized
        );


    // ========================================================
    // Cleanup
    // ========================================================

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
            d_reference
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_original_tiled
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_warp_optimized
        )
    );


    CUDA_CHECK(
        cudaFree(
            d_multiquery_optimized
        )
    );


    return result;
}


// ============================================================
// Pass Check
// ============================================================

bool is_passed(
    const ErrorMetrics& metrics,
    double mae_tolerance,
    float max_error_tolerance
)
{
    return
        metrics.mae
        <=
        mae_tolerance
        &&
        metrics.max_error
        <=
        max_error_tolerance;
}


// ============================================================
// Print One Result
// ============================================================

void print_result(
    int seq_len,
    const std::string& implementation,
    const ErrorMetrics& metrics,
    bool passed
)
{
    std::cout
        << std::left

        << std::setw(10)
        << seq_len

        << std::setw(26)
        << implementation

        << std::setw(18)
        << std::scientific
        << std::setprecision(8)
        << metrics.mae

        << std::setw(18)
        << std::scientific
        << std::setprecision(8)
        << metrics.rmse

        << std::setw(20)
        << std::scientific
        << std::setprecision(8)
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


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        constexpr int HEAD_DIM =
            64;


        // ====================================================
        // Keep original test sizes.
        //
        // We also add boundary-sensitive values for:
        //
        //     QUERIES_PER_BLOCK = 4
        //
        //     TILE_K = 32
        // ====================================================

        const std::vector<int>
            sequence_lengths =
        {
            1,
            2,
            3,
            4,
            5,

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


        constexpr double MAE_TOLERANCE =
            1e-4;


        constexpr float MAX_ERROR_TOLERANCE =
            1e-3f;


        bool all_passed =
            true;


        // ====================================================
        // Header
        // ====================================================

        std::cout
            << "\n============================================================"
            << std::endl;


        std::cout
            << "Flash Attention Correctness Regression Test"
            << std::endl;


        std::cout
            << "============================================================"
            << std::endl;


        std::cout
            << "Head Dimension: "
            << HEAD_DIM
            << std::endl;


        std::cout
            << "\nCompared Implementations:"
            << std::endl;


        std::cout
            << "1. Original Tiled Flash Attention"
            << std::endl;


        std::cout
            << "2. Warp-Optimized Tiled Flash Attention"
            << std::endl;


        std::cout
            << "3. Multi-Query Optimized Flash Attention"
            << std::endl;


        std::cout
            << std::endl;


        std::cout
            << std::left

            << std::setw(10)
            << "SeqLen"

            << std::setw(26)
            << "Flash Attention Version"

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
                104,
                '='
            )
            << std::endl;


        // ====================================================
        // Run Tests
        // ====================================================

        for(
            int seq_len :
            sequence_lengths
        )
        {
            const FlashAttentionTestResult result =
                run_test(
                    seq_len,
                    HEAD_DIM
                );


            const bool original_passed =
                is_passed(
                    result.original_tiled,
                    MAE_TOLERANCE,
                    MAX_ERROR_TOLERANCE
                );


            const bool warp_passed =
                is_passed(
                    result.warp_optimized,
                    MAE_TOLERANCE,
                    MAX_ERROR_TOLERANCE
                );


            const bool multiquery_passed =
                is_passed(
                    result.multiquery_optimized,
                    MAE_TOLERANCE,
                    MAX_ERROR_TOLERANCE
                );


            all_passed =
                all_passed
                &&
                original_passed
                &&
                warp_passed
                &&
                multiquery_passed;


            // =================================================
            // Original Tiled
            // =================================================

            print_result(
                seq_len,
                "Original Tiled",
                result.original_tiled,
                original_passed
            );


            // =================================================
            // Warp Optimized
            // =================================================

            print_result(
                seq_len,
                "Warp Optimized",
                result.warp_optimized,
                warp_passed
            );


            // =================================================
            // Multi-Query Optimized
            // =================================================

            print_result(
                seq_len,
                "MultiQuery Optimized",
                result.multiquery_optimized,
                multiquery_passed
            );


            std::cout
                << std::string(
                    104,
                    '-'
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
                << "ALL FLASH ATTENTION CORRECTNESS TESTS PASSED."
                << std::endl;


            std::cout
                << "============================================================"
                << std::endl;


            return 0;
        }


        std::cerr
            << "FLASH ATTENTION CORRECTNESS TEST FAILED."
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