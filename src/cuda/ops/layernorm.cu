#include "cuda/cuda_ops.cuh"

#include <cuda_runtime.h>
#include <math.h>

__global__
void layer_norm_kernel(const float * input, const float* gamma, const float* beta, float* output, int size, float epsilon){
    extern __shared__ float shared[];
    const int tid = threadIdx.x;


    // Threads caculate a partial sum
    float local_sum = 0.0f;

    for(int i = tid; i < size; i += blockDim.x)
        local_sum += input[i];

    shared[tid] = local_sum;

    __syncthreads();


    // Calculate total sum and mean.
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    const float mean = shared[0] / static_cast<float>(size);



    // Each threads calculates partial variance.
    float local_variance = 0.0f;

    for(int i = tid; i < size; i += blockDim.x){
        const float diff = input[i] - mean;
        local_variance += diff * diff;
    }

    shared[tid] = local_variance;

    __syncthreads();



    // Reduce partial variance values.
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }


    const float variance = shared[0] / static_cast<float>(size);



    // normalization factor
    const float norm_factor = 1.0f / sqrtf(variance + epsilon);


    // Sale and shift
    for(int i = tid; i < size; i += blockDim.x){
        const float normalized = (input[i] - mean) * norm_factor;
        output[i] = gamma[i] * normalized + beta[i];
    }


}


void cuda_layer_norm(const float* input, const float * gamma, const float* beta, float* output, int size, float epsilon, cudaStream_t stream){
    const int block_size = 256;
    const std::size_t shared_bytes = block_size * sizeof(float);

    layer_norm_kernel <<< 1, block_size, shared_bytes, stream >>> (input, gamma, beta, output, size, epsilon);
}