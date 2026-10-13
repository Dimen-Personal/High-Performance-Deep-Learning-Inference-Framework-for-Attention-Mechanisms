#include "cuda/cuda_ops.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>



// Block Max Reduction
__device__       // 只能在GPU代码内部调用的辅助函数
float block_reduce_max(float value, float* shared){
    const int tid = threadIdx.x;
    
    shared[tid] = value;

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride)
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);

        __syncthreads();
    }


    return shared[0];
}




// Block Sum Reduction
__device__
float block_reduce_sum(float value, float* shared){
    const int tid = threadIdx.x;

    shared[tid] = value;

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    return shared[0];
}




// Kernel 1: Attention softmax partial max kernel
__global__     // 可以从CPU直接启动的GPU Kernel
void attention_softmax_partial_max_kernel(
    const float* scores,
    float* partial_max,    // 记录的是每一个block的最大值
    int seq_len,
    int blocks_per_row
){
    extern __shared__ float shared[]; // 包含经过压缩的一个block的数据

    const int tid = threadIdx.x;
    
    const int row  = blockIdx.x / blocks_per_row;
    const int block_in_row = blockIdx.x % blocks_per_row;

    if(row >= seq_len)
        return;

    const int row_offset = row * blocks_per_row;
    float local_max = -FLT_MAX; // 指的是当前thread负责的所有instance的Max

    

    // 这里要注意理清结构。得到的只是
    for(int col = block_in_row * blockDim.x + tid; col < seq_len; col += blocks_per_row * blockDim.x)
        local_max = fmaxf(local_max, scores[row_offset + col]);


    const float block_max = block_reduce_max(local_max, shared);


    if(tid == 0) //等于几不重要，就是避免重复运算。
        partial_max[row * blocks_per_row + block_in_row] = block_max;

}








