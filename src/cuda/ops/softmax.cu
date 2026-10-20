#include "cuda/cuda_ops.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>


__global__
void softmax_kernel(const float* input, float* output, int size){
    extern __shared__ float shared[];   // Dynamic shared memory.

    const int tid = threadIdx.x;


    // Each thread finds its maximum
    float local_max = -FLT_MAX;   // 极小负数

    for(int i = tid; i < size; i += blockDim.x)
        local_max = fmaxf(local_max, input[i]);

    shared[tid] = local_max;

    __syncthreads();


    // Reduce all local maximum values;
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride)
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);

        __syncthreads();
    }        

    const float max_value = shared[0];



    // calculate exp(x - max) and each thread's partial sum.
    float local_sum = 0.0f;

    for(int i = tid; i < size; i += blockDim.x){
        float value = expf(input[i] - max_value);
        output[i] = value;
        local_sum += value;
    }

    shared[tid] = local_sum;
    __syncthreads();




    // reduce partial sums.
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }


    // Then we normalize it mean = 0 and variance almost equals to 1
    const float total_sum = shared[0];

    for(int i = tid; i < size; i += blockDim.x)
        output[i] /= total_sum;  
}



void cuda_softmax(const float* input, float* output, int size, cudaStream_t stream){
    const int block_size = 256;
    const std::size_t shared_bytes = block_size * sizeof(float);

    softmax_kernel <<< 1, block_size, shared_bytes, stream >>> (input, output, size);
}