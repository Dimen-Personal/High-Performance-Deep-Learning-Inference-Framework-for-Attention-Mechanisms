#pragma once

#include <cuda_runtime.h>


// ============================================================
// Simplified Fused Attention
//
// Q: [seq_len, head_dim]
// K: [seq_len, head_dim]
// V: [seq_len, head_dim]
//
// output: [seq_len, head_dim]   最后得到的是各个token的经过更新后的变量
//
// Performs:
//
// scaled QK^T
// +
// causal mask
// +
// softmax
// +
// softmax * V
//
// without materializing the full N x N score/weight matrices.
// ============================================================
void cuda_flash_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream = nullptr
);





void cuda_flash_attention_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream = nullptr
);





void cuda_flash_attention_tiled_optimized(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream = nullptr
);




