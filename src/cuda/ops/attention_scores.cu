#include "cuda/cuda_ops.cuh"

#include <cuda_runtime.h>
#include <math.h>


__global__
void attention_scores_kernrl(const float* Q, const float* K, float* scores, int seq_len, int head_dim){
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;


    // outside matrix
    if(row >= seq_len || col >= seq_len)
        return;


    // Casual mask
    if(col > row){
        scores[row * seq_len + col] = -INFINITY;
        return;
    }


    // calculate attention scores;
    float sum = 0.0f;

    for(int k = 0; k < head_dim; k++)
        sum += Q[row * head_dim + k] * K[col * head_dim + k];
    
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    scores[row * seq_len + col] = sum * scale;
}


void cuda_attention_scores(const float* Q, const float* K, float* scores, int seq_len, int head_dim, cudaStream_t stream){
    dim3 block(16,16);
    dim3 grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);

    attention_scores_kernrl <<< grid, block, 0, stream >>>(Q, K, scores, seq_len, head_dim);
}