#include "cuda/flash_attention.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdexcept>


namespace {
    constexpr int TILE_K = 32;
    constexpr int BLOCK_SIZE = 128;
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
}





__global__
void flash_attention_tiled_optimized_kernel(
    const float* __restrict__ Q,   // restrict make sure QKV and output will not point to the same memory.
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len,
    int head_dim
){
    const int query_row = blockIdx.x;
    const int tid = threadIdx.x;

    if(query_row >= seq_len)
        return;

    


    // Warp information
    const int warp_id = tid / WARP_SIZE;   // which warp this thread is in 
    const int lane = tid % WARP_SIZE;      // what's the place of this thread in this warp




    // shared mamory
    extern __shared__ float shared[];
    
    float* K_tile = shared;
    float* V_tile = K_tile + TILE_K * head_dim;
    float* scores = V_tile + TILE_K * head_dim;




    // Q row cache more quick access
    __shared__ float Q_shared[BLOCK_SIZE];





    // Online softmax state
    __shared__ float online_max;
    __shared__ float online_sum;
    __shared__ float rescale_alpha;



    // Initialize
    if(tid == 0){
        online_max = -FLT_MAX;
        online_sum = 0.0f;
        rescale_alpha = 0.0f;
    }



    if(tid < head_dim)
        Q_shared[tid] = Q[query_row * head_dim + tid];



    __syncthreads();





    // Same every output dimension will be taken by one thread
    // 当前维度上的累加和
    float output_accumulator = 0.0f;
    const float scale = rsqrtf(static_cast<float>(head_dim));






    // Casual attention, query q from key 0……q
    for(int tile_start = 0; tile_start <= query_row; tile_start += TILE_K){
        
        // Same valid keys
        const int remaining = query_row - tile_start + 1;
        const int valid_keys = remaining < TILE_K ? remaining : TILE_K;
        const int tile_elements = valid_keys * head_dim;



        // same load KV all threads be used.
        for(int index = tid; index < tile_elements; index += blockDim.x){

            // 目前这个tile的第几个Key
            const int local_key = index / head_dim;
            const int d = index % head_dim;

            const int global_key = tile_start + local_key;

            K_tile[local_key * head_dim + d] = K[global_key * head_dim + d];
            V_tile[local_key * head_dim + d] = V[global_key * head_dim + d];
        }

        __syncthreads();





        // ====================================================
        // 2. Warp-cooperative QK dot products, 主要的优化方向 一个warp的32个thread一起算一个QK点积
        //
        // Previous implementation:
        //
        // 1 thread
        //     ->
        // complete Q·K dot product
        //
        // Optimized implementation:
        //
        // 1 warp
        //     ->
        // complete Q·K dot product
        //
        //
        // BLOCK_SIZE = 128
        //
        // therefore:
        //
        // 4 warps
        //
        // can calculate 4 keys concurrently.
        // ====================================================
        // Here local_key decide which key to be calculate
        // Here thread responsible to every dimension of the key
        for(int local_key = warp_id; local_key < valid_keys; local_key += WARPS_PER_BLOCK){

            // This partial sum is for every thread
            float partial_sum = 0.0f;   

            for(int d = lane; d < head_dim; d +=  WARP_SIZE)
                partial_sum += Q_shared[d] * K_tile[local_key * head_dim + d];


            // Warp reduction
            // We can use warp-level communication at here, so we don't need to use stncthreads
            for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
                partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset); // 0xffffffff全是1 表示所有的lane都参与这次shuffle， 第二个表示要取什么值，第三个表示延后多少个lane去取

            
            // 合并完之后第一个thread的partial sum就是这个warp的总和
            if(lane == 0)
                scores[local_key] = partial_sum * scale;

        }






        // Then we will use warp 0 to calculate tile maximum
        // Therefore, one lane will correspond to one score
        
        __syncthreads();    // make sure all scores been calculate.

        if(warp_id == 0){

            // 每一个lane 读取当前key的score
            float max_value = lane < valid_keys ? scores[lane] : -FLT_MAX;



            // Max reduction
            for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2){
                
                const float other = __shfl_down_sync(0xffffffff, max_value, offset);

                max_value = fmaxf(max_value, other);
            }



            // Update online maximum, choose one lane to do so;
            if(lane == 0){
                const float old_max = online_max;
                const float new_max = fmaxf(old_max, max_value);

                if(online_sum == 0.0f)
                    rescale_alpha = 0.0f;
                else
                    rescale_alpha = expf(old_max - new_max);

                online_max = new_max;
            }


            // 需要等到lane 0写完
            __syncwarp();



            // Transfer to exponential
            float beta = 0.0f;

            // 这个是应对小数据量情况
            if(lane < valid_keys){
                beta = expf(scores[lane] - online_max);

                scores[lane] = beta;
            }




            // Sum reduction
            float tile_sum = beta;

            for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);


            



            // Update online normalization sum, lane 0 will store tile sum.
            if(lane == 0)
                online_sum = rescale_alpha * online_sum + tile_sum;

        }



        // Other warps need online sum to update output.  warp = 0 正时结束，进入tile
        __syncthreads();





        // Update partial output, each threads owns one output dimension.
        if(tid < head_dim){
            
            float tile_output = 0.0f;

            for(int local_key = 0; local_key < valid_keys; local_key ++)
                tile_output += scores[local_key] * V_tile[local_key * head_dim + tid];

            output_accumulator = rescale_alpha * output_accumulator + tile_output;  //当前维度总和
        }



        __syncthreads();
            

    }



    // 这里维度总和已经更新完了，就更新最终的输出结果，除以soft Max的底数
    if(tid < head_dim)
        output[query_row * head_dim + tid] = output_accumulator / online_sum;

}





// Host Wrapper
void cuda_flash_attention_tiled_optimized(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream
){
    if(Q == nullptr || K == nullptr || V == nullptr || output == nullptr)
        throw std::invalid_argument("Optimized Flash Attention received null pointer.");


    if(seq_len <= 0 || head_dim <= 0)
        throw std::invalid_argument("Optimized Flash Attention dimensions must be positive.");


    if(head_dim > BLOCK_SIZE)
        throw std::invalid_argument("Optimized tiled flash attention currently requires head_dim <= 128.");


    

    // Dynamic shared mmory
    const std::size_t shared_floats = static_cast<std::size_t>(2 * TILE_K * head_dim + TILE_K);
    const std::size_t shared_bytes = shared_floats * sizeof(float);


    flash_attention_tiled_optimized_kernel <<< seq_len, BLOCK_SIZE, shared_bytes, stream >>> (
        Q, K, V,
        output,
        seq_len,
        head_dim
    );


}



