// 标准 3- kernel baseline vs. tiled online-softmax
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

    float max_error =
        0.0f;
};


// ============================================================
// Deterministic initialization
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
// Error calculation
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


        const double absolute =
            std::fabs(
                difference
            );


        sum_absolute +=
            absolute;


        sum_squared +=
            difference
            *
            difference;


        metrics.max_error =
            std::max(
                metrics.max_error,
                static_cast<float>(
                    absolute
                )
            );
    }


    metrics.mae =
        sum_absolute
        /
        reference.size();


    metrics.rmse =
        std::sqrt(
            sum_squared
            /
            reference.size()
        );


    return metrics;
}


// ============================================================
// One test
// ============================================================

ErrorMetrics run_test(
    int seq_len,
    int head_dim
)
{
    const std::size_t qkv_elements =
        static_cast<std::size_t>(
            seq_len
        )
        *
        head_dim;


    const std::size_t attention_elements =
        static_cast<std::size_t>(
            seq_len
        )
        *
        seq_len;


    // ========================================================
    // Host
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


    std::vector<float> h_flash(
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
    // Device
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
            &d_flash,
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
    // Tiled online-softmax Flash Attention
    // ========================================================

    cuda_flash_attention_tiled(
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
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            h_flash.data(),
            d_flash,
            qkv_elements
            *
            sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );


    const ErrorMetrics metrics =
        calculate_error(
            h_reference,
            h_flash
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
            256,
            512
        };


        // Online softmax changes floating-point operation order
        // more substantially than the naive fused kernel.
        //
        // Therefore use a slightly more relaxed tolerance.
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
            << "Tiled Flash Attention Correctness Test"
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
                    metrics.max_error
                    <=
                    MAX_ERROR_TOLERANCE
                );


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


        std::cout
            << "\n============================================"
            << std::endl;


        if(all_passed)
        {
            std::cout
                << "ALL TILED FLASH ATTENTION TESTS PASSED."
                << std::endl;


            std::cout
                << "============================================"
                << std::endl;


            return 0;
        }


        std::cerr
            << "TILED FLASH ATTENTION TEST FAILED."
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