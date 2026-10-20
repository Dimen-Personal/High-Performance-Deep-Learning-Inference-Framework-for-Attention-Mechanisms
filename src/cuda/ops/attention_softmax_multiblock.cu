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

    const int row_offset = row * seq_len;
    float local_max = -FLT_MAX; // 指的是当前thread负责的所有instance的Max

    

    // 这里要注意理清结构。得到的只是
    for(int col = block_in_row * blockDim.x + tid; col < seq_len; col += blocks_per_row * blockDim.x)
        local_max = fmaxf(local_max, scores[row_offset + col]);


    const float block_max = block_reduce_max(local_max, shared);


    if(tid == 0) //等于几不重要，就是避免重复运算。
        partial_max[row * blocks_per_row + block_in_row] = block_max;

}


// Kernel2: row max
// 注意这边是一个block去处理partial score里的一行
__global__
void attention_softmax_row_max_kernel(
    const float* partial_max,
    float* row_max,
    int seq_len,
    int blocks_per_row
){
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if(row >= seq_len)
        return;


    float local_max = -FLT_MAX; // 这里也是一样，当前thread的最大值
    for(int i = tid; i < blocks_per_row; i += blockDim.x)
        local_max = fmaxf(local_max, partial_max[row * blocks_per_row + i]);


    const float max_value = block_reduce_max(local_max, shared);

    if(tid == 0)
        row_max[row] = max_value;
}



// Kernel 3: Calculate Partial block sum and exp.
__global__
void attention_softmax_exp_sum_kernel(
    const float* scores,
    float* weights,
    const float* row_max,
    float* partial_sum,
    int seq_len,
    int blocks_per_row
){
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int row = blockIdx.x / blocks_per_row;
    const int block_in_row = blockIdx.x % blocks_per_row;

    if(row >= seq_len) return;

    const int row_offset = row * seq_len;
    const float max_value = row_max[row];

    float local_sum = 0.0f;



    for(int col = block_in_row * blockDim.x + tid; col < seq_len; col += blockDim.x * blocks_per_row){
        const float value = expf(scores[row_offset + col] - max_value);

        weights[row_offset + col] = value;

        local_sum += value;
    }


    const float block_sum = block_reduce_sum(local_sum, shared);

    if(tid == 0)
        partial_sum[row * blocks_per_row + block_in_row] = block_sum;
}





// Kernel 4: Calculate row sum.
__global__
void attention_softmax_row_sum_kernel(
    const float* partial_sum,
    float* row_sum,
    int seq_len,
    int blocks_per_row
){
    extern __shared__ float shared[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if(row >= seq_len)
        return;


    float local_sum = 0.0f;

    for(int i = tid; i < blocks_per_row; i += blockDim.x)
        local_sum += partial_sum[row * blocks_per_row + i];


    const float total_sum = block_reduce_sum(local_sum, shared);

    if(tid == 0)
        row_sum[row] = total_sum;
}




// Kernel 5: normalize
__global__
void attention_softmax_normalize_kernel(
    float* weights,
    const float* row_sum,
    int seq_len,
    int blocks_per_row
){
    const int tid = threadIdx.x;
    const int row = blockIdx.x / blocks_per_row;
    const int block_in_row = blockIdx.x % blocks_per_row;

    if(row >= seq_len)
        return;

    const int row_offset = row * seq_len;
    const float total_sum = row_sum[row];


    for(int col = block_in_row * blockDim.x + tid; col < seq_len; col += blockDim.x * blocks_per_row)
        weights[row_offset + col] /= total_sum;
}






// Final Host function
void cuda_attention_softmax_multiblock(
    const float* scores,
    float* weights,
    float* partial_max,
    float* partial_sum,
    float* row_max,
    float* row_sum,
    int seq_len,
    int blocks_per_row,
    cudaStream_t stream
){
    const int block_size = 256; 
    const std::size_t shared_bytes = block_size * sizeof(float);
    const int partial_grid = seq_len * blocks_per_row;

    // Partial Max
    attention_softmax_partial_max_kernel <<< partial_grid, block_size, shared_bytes, stream >>>(
        scores,
        partial_max,
        seq_len, 
        blocks_per_row
    );


    // Final maximum of each row
    attention_softmax_row_max_kernel <<< seq_len, block_size, shared_bytes, stream >>>(
        partial_max, 
        row_max,
        seq_len,
        blocks_per_row
    );



    // Exp + partial sum
    attention_softmax_exp_sum_kernel <<< partial_grid, block_size, shared_bytes, stream >>>(
        scores,
        weights,
        row_max,
        partial_sum,
        seq_len,
        blocks_per_row
    );



    // complete row sum
    attention_softmax_row_sum_kernel <<< seq_len, block_size, shared_bytes, stream >>>(
        partial_sum,
        row_sum,
        seq_len,
        blocks_per_row
    );




    // Normalize
    attention_softmax_normalize_kernel <<< partial_grid, block_size, 0, stream >>>(
        weights,
        row_sum,
        seq_len,
        blocks_per_row
    );

}



