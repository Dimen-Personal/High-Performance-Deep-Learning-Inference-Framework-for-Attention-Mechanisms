#include "cuda/flash_attention.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdexcept>


// Configuration
namespace{
    constexpr int TILE_K = 32;
    constexpr int BLOCK_SIZE = 128;
}






// ========================================================
    // Dynamic shared memory layout
    //
    // K_tile:
    //     TILE_K × head_dim
    //
    // V_tile:
    //     TILE_K × head_dim
    //
    // scores:
    //     TILE_K
    //
    // reduction:
    //     TILE_K
    //
    // Total:
    //
    // 2 * TILE_K * head_dim
    // +
    // 2 * TILE_K
    //
    // floats
// ========================================================

__global__
void flash_attention_tiled_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim
){
    const int query_row = blockIdx.x;     // One block represents one query
    const int tid = threadIdx.x;

    if(query_row >= seq_len)
        return;




    // Data Initialization
    extern __shared__ float shared[];


    float* K_tile = shared;
    float* V_tile = K_tile + TILE_K * head_dim;
    float* scores = V_tile + TILE_K * head_dim;

    float* reduction = scores + TILE_K;



    // Online softmax state, shared between all threads
    __shared__ float online_max;
    __shared__ float online_sum;
    __shared__ float rescale_alpha;


    if(tid == 0){
        online_max = -FLT_MAX;
        online_sum = 0.0f;
        rescale_alpha = 0.0f;
    }


    __syncthreads();


    // Due to we use one block to represent one query, threrfore, head_dim <= Block_size.
    // 每个 thread 在输出纬度的确认上还是拥有一个output的输出纬度
    float output_accumulator = 0.0f;


    const float scale = rsqrtf(static_cast<float>(head_dim));



    // We have casual mask at here.
    // Now we calculate every tile one by one
    for(int tile_start = 0; tile_start <= query_row; tile_start += TILE_K){

        // Calculate valid elements number on each tile.
        const int remaining = query_row - tile_start + 1;
        const int valid_keys = remaining < TILE_K ? remaining : TILE_K;
        const int tile_elements = valid_keys * head_dim;



        // 1. Load K tile and V tile into shared memory.
        // 这边由每个thread共同完成搬运。每一个 thread 去搬一个 数字
        for(int index = tid; index < tile_elements; index += blockDim.x){

            // 目前这个tile的第几个Key
            const int local_key = index / head_dim;
            const int d = index % head_dim;

            const int global_key = tile_start + local_key;

            K_tile[local_key * head_dim + d] = K[global_key * head_dim + d];
            V_tile[local_key * head_dim + d] = V[global_key * head_dim + d];
        }

        __syncthreads();



        // 2. Compute scores for every key in this tile.
        // One threads responsible for one Q * K dot product.
        // 后面其中的一个优化方向也是可以通过多个thread去处理一个QV Product
        if(tid < TILE_K){

            if(tid < valid_keys){
                float score = 0.0f;

                for(int d = 0; d < head_dim; d++)
                    score += Q[query_row * head_dim + d] * K_tile[tid * head_dim + d];

                scores[tid] = score * scale;

            } else{
                scores[tid] = -FLT_MAX;
            }
        }



        __syncthreads();




        // 3. Find the maximum on the current tile (we have already got all the scores on this tile).
        if(tid < TILE_K)
            reduction[tid] = scores[tid];


        __syncthreads();



        for(int stride = TILE_K / 2; stride > 0; stride /= 2){
            
            if(tid < stride)
                reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);

            __syncthreads();
        }


        const float tile_max = reduction[0];






        // 4. Merge previous with new tile maximum
        // Point thread 0 to calculate new max and the rescaling factor
        // notice we only need to keep max, sum and exp(old max - new max) * VA 就行
        if(tid == 0){

            const float old_max = online_max;
            const float new_max = fmaxf(old_max, tile_max);


            if(online_sum == 0.0f)
                rescale_alpha = 0.0f;
            else 
                rescale_alpha = expf(old_max - new_max);

            online_max = new_max;
        }

        __syncthreads();






        
        // 5. Convert tile scores into exp(score - new_max)
        // Also calculate: tile_sum
        // 还是那32个thread 去负责
        if(tid < TILE_K){
            
            if(tid < valid_keys){
                const float beta = expf(scores[tid] - online_max);

                scores[tid] = beta;
                reduction[tid] = beta;

            }else{
                scores[tid] = 0.0f;
                
                reduction[tid] = 0.0f;
            }
        }


        __syncthreads();



        // 同时计算这个tile sum.
        for(int stride = TILE_K / 2; stride > 0; stride /= 2){
            
            if(tid < stride)
                reduction[tid] += reduction[tid + stride];

            __syncthreads();
        }


        const float tile_sum = reduction[0];







        // 6. Update online normalization sum
        // l_new = alpha * l_old + sum(beta)
        if(tid == 0)
            online_sum = rescale_alpha * online_sum + tile_sum;

        __syncthreads();






        // 7. Update partial output:
        // O_new = alpha * O_old + sum(beta_j * V_j)
        // Each thread owns one dimention
        if(tid < head_dim){

            float tile_output = 0.0f;

            for(int local_key = 0; local_key < valid_keys; local_key ++)
                tile_output += scores[local_key] * V_tile[local_key * head_dim + tid];

            
            output_accumulator = rescale_alpha * output_accumulator + tile_output;
            
        }




        // *** ensure every thread is finished by using this tile.
        __syncthreads();


    }






    // Final Normalization
    if(tid < head_dim)
        output[query_row * head_dim + tid] = output_accumulator / online_sum;

}







// Host Wrapper
void cuda_flash_attention_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    cudaStream_t stream
){
    if(Q == nullptr || K == nullptr || V == nullptr || output == nullptr)
        throw std::invalid_argument("Flash attention received null pointer.");


    if(seq_len <= 0 || head_dim <= 0)
        throw std::invalid_argument("Flash Attention dimensions must be positive.");

    if(head_dim > BLOCK_SIZE)
        throw std::invalid_argument("Cuurrent tiled flash attention requires head_dim <= 128.");



    // Dynamic shared memory requirement
    // K_tile: TILE_K * head_dim
    // V_tile: TILE_K * head_dim
    // scores: Tile_K
    // reduction: Tile_K
    const std::size_t shared_floats = static_cast<std::size_t>(2*TILE_K*head_dim + 2 * TILE_K);

    const std::size_t shared_bytes = shared_floats * sizeof(float);


    flash_attention_tiled_kernel <<< seq_len, BLOCK_SIZE, shared_bytes, stream >>>(
        Q, K, V,
        output,
        seq_len,
        head_dim
    );
}





