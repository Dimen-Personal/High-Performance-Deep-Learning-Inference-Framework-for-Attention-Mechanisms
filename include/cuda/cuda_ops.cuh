#ifndef CUDA_OPS_CUH
#define CUDA_OPS_CUH

#include <cuda_runtime.h>

void cuda_vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
);


void cuda_gelu(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = nullptr    // 这里nullptr 相当于默认的stream
);


void cuda_softmax(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream = nullptr
);

void cuda_layer_norm(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int size,
    float epsilon,
    cudaStream_t stream = nullptr
);


void cuda_matmul(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
    cudaStream_t stream = nullptr
);


void cuda_attention_scores(
    const float* Q,
    const float* K,
    float* scores,
    int seq_len,
    int head_dim,
    cudaStream_t stream = nullptr
);


void cuda_attention_softmax(
    const float* scores,
    float* weights,
    int seq_len,
    cudaStream_t stream = nullptr
);

// Second version of the cuda_attention_softmax
// Second version of the cuda_attention_softmax
void cuda_attention_softmax_multiblock(
    const float* scores,
    float* weights,
    float* partial_max,   // Allocate space for the SM communication prior to the actual execution.
    float* partial_sum,
    float* row_max,
    float* row_sum,
    int seq_len,
    int blocks_per_row,    // Decide use how many blocks to deal with a row.
    cudaStream_t stream = nullptr
);

#endif