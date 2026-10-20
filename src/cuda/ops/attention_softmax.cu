#include "cuda/cuda_ops.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>


__global__
void attention_softmax_kernel(const float* scores, float* weights, int seq_len){
    extern __shared__ float shared[]; // extern = dynamic, __shared__ one block shared memory.

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if(row >= seq_len)
        return;

    const int row_offset = row * seq_len;




    // Find row maximum
    float local_max = -FLT_MAX;

    for(int col = tid; col < seq_len; col += blockDim.x)
        local_max = fmaxf(local_max, scores[row_offset + col]);

    shared[tid] = local_max;

    __syncthreads();



    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride) 
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);

        __syncthreads();
    }


    const float max_value = shared[0];






    // Try to calculate total sum.
    float local_sum = 0.0f;

    for(int col = tid; col < seq_len; col += blockDim.x){
        const float value = expf(scores[row_offset + col] - max_value);
        weights[row_offset + col] = value;

        local_sum += value;
    }

    shared[tid] = local_sum;

    __syncthreads();





    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }



    const float total_sum = shared[0];




    // final normalize row.
    for(int col = tid; col < seq_len; col += blockDim.x)
        weights[row_offset + col] /= total_sum;

}




void cuda_attention_softmax(const float* scores, float* weights, int seq_len, cudaStream_t stream){
    const int block_size = 256;
    const std::size_t shared_bytes = block_size * sizeof(float);


    attention_softmax_kernel <<< seq_len, block_size, shared_bytes, stream >>> (scores, weights, seq_len);
}