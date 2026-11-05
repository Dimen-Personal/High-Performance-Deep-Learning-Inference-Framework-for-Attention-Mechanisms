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
                +                                                            \
                cudaGetErrorString(error)                                    \
            );                                                               \
        }                                                                    \
    } while(false)


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
};


// ============================================================
// Deterministic initialization
//
// Keep values relatively small so the attention logits
// remain numerically reasonable.
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
// Calculate output error
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
            "Reference/output size mismatch."
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


        absolute_error_sum +=
            absolute_error;


        squared_error_sum +=
            difference
            *
            difference;


        metrics.max_absolute_error =
            std::max(
                metrics.max_absolute_error,
                static_cast<float>(
                    absolute_error
                )
            );
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


    return metrics;
}


// ============================================================
// Run one sequence length
// ============================================================

ErrorMetrics run_test(
    int seq_len,
    int head_dim
)
{
    // ========================================================
    // Tensor sizes
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


    const std::size_t output_elements =
        qkv_elements;


    // ========================================================
    // Host buffers
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
        output_elements
    );


    std::vector<float> h_flash(
        output_elements
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


    float* d_reference =
        nullptr;


    float* d_flash =
        nullptr;


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_Q
            ),
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_K
            ),
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_V
            ),
            qkv_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_scores
            ),
            attention_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_weights
            ),
            attention_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_reference
            ),
            output_elements
            *
            sizeof(float)
        )
    );


    CUDA_CHECK(
        cudaMalloc(
            reinterpret_cast<void**>(
                &d_flash
            ),
            output_elements
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
    // Baseline:
    //
    // AttentionScores
    //      ↓
    // Softmax
    //      ↓
    // MatMul(weights, V)
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


    // --------------------------------------------------------
    // weights:
    //     [seq_len, seq_len]
    //
    // V:
    //     [seq_len, head_dim]
    //
    // output:
    //     [seq_len, head_dim]
    //
    // M = seq_len
    // K = seq_len
    // N = head_dim
    // --------------------------------------------------------

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


    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ========================================================
    // Fused Attention
    // ========================================================

    cuda_flash_attention(
        d_Q,
        d_K,
        d_V,
        d_flash,
        seq_len,
        head_dim
    );


    CUDA_CHECK(
        cudaGetLastError()
    );


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
            output_elements
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_flash.data(),
            d_flash,
            output_elements
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // Calculate error
    // ========================================================

    const ErrorMetrics metrics =
        calculate_error_metrics(
            h_reference,
            h_flash
        );


    // ========================================================
    // Optional:
    // print first output row for easier debugging
    // ========================================================

    std::cout
        << "\nFirst 8 values of output[0]:"
        << std::endl;


    std::cout
        << std::left
        << std::setw(8)
        << "Dim"
        << std::setw(18)
        << "Baseline"
        << std::setw(18)
        << "Flash"
        << std::setw(18)
        << "Abs Error"
        << std::endl;


    std::cout
        << std::string(
            62,
            '-'
        )
        << std::endl;


    const int print_count =
        std::min(
            8,
            head_dim
        );


    for(
        int d = 0;
        d < print_count;
        d++
    )
    {
        const float error =
            std::fabs(
                h_reference[d]
                -
                h_flash[d]
            );


        std::cout
            << std::left
            << std::setw(8)
            << d
            << std::setw(18)
            << std::fixed
            << std::setprecision(8)
            << h_reference[d]
            << std::setw(18)
            << std::fixed
            << std::setprecision(8)
            << h_flash[d]
            << std::setw(18)
            << std::fixed
            << std::setprecision(8)
            << error
            << std::endl;
    }


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
            d_flash
        )
    );


    return metrics;
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


        const std::vector<int>
            sequence_lengths =
        {
            32,
            64,
            128,
            256
        };


        // ====================================================
        // Because the two paths perform mathematically
        // equivalent FP32 operations in different orders,
        // exact bitwise equality is not required.
        // ====================================================

        constexpr double MAE_TOLERANCE =
            1e-4;


        constexpr float MAX_ERROR_TOLERANCE =
            1e-3f;


        bool all_passed =
            true;


        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "Flash Attention Correctness Test"
            << std::endl;


        std::cout
            << "============================================"
            << std::endl;


        std::cout
            << "Head Dimension: "
            << HEAD_DIM
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


        for(
            int seq_len :
            sequence_lengths
        )
        {
            std::cout
                << "\nTesting seq_len = "
                << seq_len
                << "..."
                << std::endl;


            const ErrorMetrics metrics =
                run_test(
                    seq_len,
                    HEAD_DIM
                );


            const bool passed =
                (
                    metrics.mae
                    <=
                    MAE_TOLERANCE
                )
                &&
                (
                    metrics.max_absolute_error
                    <=
                    MAX_ERROR_TOLERANCE
                );


            if(!passed)
            {
                all_passed =
                    false;
            }


            std::cout
                << "\nSummary:"
                << std::endl;


            std::cout
                << std::left
                << std::setw(12)
                << seq_len
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
                << metrics.max_absolute_error
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
            << "\n============================================"
            << std::endl;


        if(all_passed)
        {
            std::cout
                << "ALL FLASH ATTENTION CORRECTNESS TESTS PASSED."
                << std::endl;


            std::cout
                << "============================================"
                << std::endl;


            return 0;
        }


        std::cerr
            << "FLASH ATTENTION CORRECTNESS TEST FAILED."
            << std::endl;


        std::cerr
            << "============================================"
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