#include "cuda/cuda_ops.cuh"

#include <cuda_runtime.h>
#include <math.h>


__global__ void gelu_kernel(const float* input, float* output, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= size) return;

    const float div_sqrt_2 = 1.0f / sqrtf(2.0f);

    float x = input[i];
    output[i] = 0.5f * x * (1.0f + erff(x * div_sqrt_2));
}


void cuda_gelu(const float* input, float* output, int size, cudaStream_t stream){
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    gelu_kernel<<< grid_size, block_size, 0, stream >>>(input, output, size);    // Reprsents 0 shared bytes and stream.

}