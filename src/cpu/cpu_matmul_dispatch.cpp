#include "cpu/cpu_matmul_dispatch.h"

#include "cpu/fp32_matmul.h"

#include "quantization/int8_matmul.h"
#include "quantization/quantization.h"

#include <cstddef>
#include <stdexcept>


void cpu_matmul_dispatch(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
    const CpuMatmulConfig& config
)
{
    // ========================================================
    // Validation
    // ========================================================

    if(A == nullptr)
    {
        throw std::invalid_argument(
            "CPU MatMul A is null."
        );
    }


    if(B == nullptr)
    {
        throw std::invalid_argument(
            "CPU MatMul B is null."
        );
    }


    if(C == nullptr)
    {
        throw std::invalid_argument(
            "CPU MatMul C is null."
        );
    }


    if(
        M <= 0
        ||
        K <= 0
        ||
        N <= 0
    )
    {
        throw std::invalid_argument(
            "CPU MatMul dimensions must be positive."
        );
    }


    if(config.thread_count <= 0)
    {
        throw std::invalid_argument(
            "CPU MatMul thread count must be positive."
        );
    }


    // ========================================================
    // FP32 execution path
    // ========================================================

    if(
        config.mode
        ==
        CpuMatmulMode::FP32
    )
    {
        fp32_matmul_openmp(
            A,
            B,
            C,
            M,
            K,
            N,
            config.thread_count
        );


        return;
    }


    // ========================================================
    // INT8 execution path
    //
    // Current implementation performs dynamic per-tensor
    // symmetric quantization.
    //
    // Later, weights B can be pre-quantized and cached.
    // ========================================================

    if(
        config.mode
        ==
        CpuMatmulMode::INT8
    )
    {
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


        // ----------------------------------------------------
        // FP32 → INT8
        // ----------------------------------------------------

        const QuantizedTensor A_quantized =
            quantize_fp32_to_int8(
                A,
                A_size
            );


        const QuantizedTensor B_quantized =
            quantize_fp32_to_int8(
                B,
                B_size
            );


        // ----------------------------------------------------
        // INT8 OpenMP MatMul
        // ----------------------------------------------------

        int8_matmul_openmp(
            A_quantized.data.data(),
            B_quantized.data.data(),
            C,
            M,
            K,
            N,
            A_quantized.params,
            B_quantized.params,
            config.thread_count
        );


        return;
    }


    throw std::runtime_error(
        "Unsupported CPU MatMul execution mode."
    );
}