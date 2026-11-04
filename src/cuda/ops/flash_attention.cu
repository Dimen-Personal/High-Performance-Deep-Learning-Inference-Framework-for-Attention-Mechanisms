#include "cuda/flash_attention.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>


__global__
void flash_attention_naive_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim
){
    // One block kernel handles one query row
    const int query_row = blockIdx.x;
    const int tid = threadIdx.x;

    if(query_row > seq_len)
        return;

    
    // This template version still stores onr row on the shared memory instead N * N matrix in HBM
    extern __shared__ float shared[];
    float* scores = shared;
    float* reduction = socres + seq_len;



    // Scale
    const float scale = rsqrtf(static_cast<float>(head_dim));



    // step 1: attention scores, shared_memory stores 
    // Query 行和 key 列都已经确定了
    for(int key_col = tid; key_col <= query_row; key_col += blockDim.x){
        
        float score = 0.0f;

        for(int d = 0; d < head_dim; d++)
            scores += Q[query_row * head_dim + d] * K[key_col * head_dim + d];
        
        scores[key_col] = score * scale;
    }


    __syncthreads();





    // 2. find maximum score.
    float local_max = -FLT_MAX;

    for(int key_col = tid; key_col <= query_row; key_col += blockDim.x)
        local_max = fmaxf(local_max, scores[key_col]);

    reduction[tid] = local_max;

    __synthreads();


    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        
        if(tid < stride)
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);

        __synthreads();
    }


    const float row_max = reduction[0];




    // 3. Calculate exp(score - max) and calculate row exp sum
    float local_sum = 0.0f;

    for(int key_col = tid; key_col <= query_row; key_col += blockDim.x){
        const float weight = expf(scores[key_col] - row_max);

        scores[key_col] = weight;

        local_sum += weight;
    }


    reduction[tid] = local_sum;

    __synthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride)
            reduction[tid] += reduction[tid + stride];


        __synthreads();
    }



    const float row_sum = reduction[0];


    __synthreads();





    // 4. Compute final output[row, D], output[row] 存储的是当前query代表的token最后的加权值，output[row,d]代表这个加权值在纬度d上的值
    for(int d = tid; d < head_dim; d += blockDim.x){
        
        float result = 0.0f;

        for(int key_col = 0; key_col <= query_row; key_col ++){
            
            const float weight = scores[key_col] / row_sum;

            result += weight * V[key_col * head_dim + d]; 
        }

        
        output[query_row * head_dim + d] = result;
    }

}



// Host wrapper (still O (N) storage)
void cuda_flash_atention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream
){
    constexpr int BLOCK_SIZE = 256;
    const int grid_size = seq_len;

    // seq_len used to store scores of currect query and other's key, and block size to be used for reduction.
    const std::size_t shared_bytes = static_cast<std::size_t>(seq_len + BLOCK_SIZE) * sizeof(float);

    flash_attention_naive_kernel <<< grid_size, BLOCK_SIZE, shared_bytes, stream >>>(
        Q, K, V,
        output,
        seq_len, head_dim
    );
}