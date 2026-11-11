#include "cuda/flash_attention.cuh"

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdexcept>



namespace{
    constexpr int WARP_SIZE = 32;
    constexpr int BLOCK_SIZE = 128;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    constexpr int QUERYS_PER_BLOCK = WARPS_PER_BLOCK;

    constexpr int TILE_K = 32;

    // curretn we need to make sure head_dim < 128
    // Therefore each lane may at most have 4 dimention needed to be deal
    constexpr int MAX_DIMS_PER_LANE = 4;
}





// ============================================================
// Multi-Query Tiled Flash Attention
//
// Mapping:
//
//     1 block = 4 queries
//
//     warp 0 -> query 0
//     warp 1 -> query 1
//     warp 2 -> query 2
//     warp 3 -> query 3
//
// Main idea:
//
//     K/V tile is loaded ONCE
//
// and reused by all four query warps.
//
// This improves on the previous:
//
//     1 block = 1 query
//
// structure, where every query block independently loaded the
// same K/V tiles from global memory.
// ============================================================
__global__
void flash_attention_multiquery_optimized_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int seq_len,
    int head_dim
){
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;


    // Query range handle by this block
    const int query_start = blockIdx.x * QUERYS_PER_BLOCK;
    const int query_row = query_start + warp_id;
    const bool query_active = query_row < seq_len;

    int max_query = query_start + QUERYS_PER_BLOCK - 1;
    if(max_query >= seq_len)
        max_query = seq_len - 1;




    // ========================================================
    // Dynamic shared-memory layout
    //
    // K_tile:
    //
    //     TILE_K × head_dim
    //
    // V_tile:
    //
    //     TILE_K × head_dim
    //
    // Q_tile:
    //
    //     QUERIES_PER_BLOCK × head_dim
    //
    // scores:
    //
    //     QUERIES_PER_BLOCK × TILE_K
    //
    //
    // Most importantly:
    //
    // K_tile and V_tile are shared by all four query warps.
    // ========================================================
    extern __shared__ float shared[];

    float* K_tile = shared;
    float* V_tile = K_tile + TILE_K * head_dim;
    float* Q_tile = V_tile + TILE_K * head_dim;
    float* scores = Q_tile + QUERYS_PER_BLOCK * head_dim;






    // Load Q rows once, all threads cooperate
    const int total_q_elements = QUERYS_PER_BLOCK * head_dim;

    for(int index = tid; index < total_q_elements; index += blockDim.x){
        const int local_query = index / head_dim;
        const int d = index % head_dim;

        const int global_query = query_start + local_query;

        if(global_query < seq_len)
            Q_tile[local_query * head_dim + d] = Q[global_query * head_dim + d];
        else
            Q_tile[local_query * head_dim + d] = 0.0f;

    }


    __syncthreads();







    // Each warp maintains its own online-softmax state.
    //
    // These values are kept in registers.
    //
    // All lanes inside the same warp keep identical copies.
    float online_max = -FLT_MAX;
    float online_sum = 0.0f;






    // Output accumulators.
    // One warp handles one query.
    float output_accumulator[MAX_DIMS_PER_LANE]

    #pragma unroll    // 尽量把循环直接展开，因为数据量实际上很小。
    for(int i = 0; i < MAX_DIMS_PER_LANE; i++)
        output_accumulator[i] = 0.0f;

    const float scale = rsqrtf(static_cast<float>(head_dim));







    // Process KV tile, all queries in this CTA share the same KV tile.
    // And since we do the casual mask, so the fartest will be max Q query number this block handles
    for(int tile_start = 0; tile_start <= max_query; tile_start += TILE_K){

        // Load number of K/V rows
        const int remaining_keys = max_query - tile_start + 1;
        const int tile_key_count = remaining_keys < TILE_K ? remaining_keys : TILE_K;
        const int tile_elements = tile_key_count * head_dim;


        // 1. Load this K/V tile
        // All four query warps reuse this tile.
        for(int index = tid; index < tile_elements; index += blockDim.x){
            const int local_key = index / head_dim;
            const int d = index % head_dim;

            const int global_key = tile_start + local_key;

            K_tile[local_key * head_dim + d] = K[global_key * head_dim + d];
            V_tile[local_key * head_dim + d] = V[global_key * head_dim + d];
        }




        // Every warp need this KV tile.
        __syncthreads();




        // ====================================================
        // Number of keys THIS query can attend to.
        //
        // Example:
        //
        // tile_start = 32
        //
        // query = 33
        //
        // valid keys in tile:
        //
        //     key 32
        //     key 33
        //
        // so query_valid_keys = 2.
        // ====================================================
        int query_valid_keys = 0;


        if(query_active && query_row >= tile_start){
            query_valid_keys = query_row - tile_start + 1;


            // 这个是query远远大于当前tile的情况
            if(query_valid_keys > tile_key_count)
                query_valid_keys = tile_key_count;
        }






        // 2. each warp calculates scores for its own query.
        // Entire warp coolane cooperatively calculates each QK
        if(query_active){
            
            for(int local_key = 0; local_key < query_valid_keys; local_key ++){

                float partial_sum = 0.0f;

                for(int d = lane; d < head_dim; d += WARP_SIZE)
                    partial_sum += Q_tile[warp_id * head_dim + d] * K_tile[local_key * head_dim + d];



                // warp reduction
                for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
                    partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);

                if(lane == 0)
                    scores[warp_id * TILE_K + local_key] = partia_sum * scale;
            }
        }




        // Only threads inside the same warp need the scores written by lane 0.
        __syncwarp();




        // Queries before this K tile have zero valid keys so just simply skip it. don't needed to be updated.
        if(query_active && query_valid_keys > 0){
             
            // 3. Find the maximum
            // Each lane reads one score.
            float tile_max = lane < query_valid_keys ?  scores[warp_id * TILE_K + lane] : -FLT_MAX;

            for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2){
                const float other = __shfl_down_sync(0xffffffff, tile_max, offset);

                tile_max = fmaxf(tile_max, other);
            }


            // Let all lane on the same warp knows what's the max
            tile_max = __shfl_sync(0xffffffff, tile_max, 0);



            // 4. online-softmax maximum merge
            const float new_max = fmaxf(online_max, tile_max);

            const float alpha = online_sum == 0.0f ? 0.0f : expf(online_max - new_max);



            // 5. compute exp
            float beta = 0.0f;
            if(lane < query_valid_keys){
                beta = expf(scores[warp_id * TILE_K + lane] - new_max);

                scores[warp_id * TILE_K + lane] = beta;
            }


            // All exp writes.
            __syncwarp();





            // 6. Tile sum
            float tile_sum = beta;

            for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
                tile_sum += __shfl_down_sync(0xffffffff, tile_sum, offset);

            tile_sum = __shfl_sync(0xffffffff, tile_sum, 0);





            // 7. Update output numerator.
            // Each lane owns up to 4 output dimensions.
            #pragma unroll
            for(int output_slot = 0; output_slot < MAX_DIMS_PER_LANE; output_slot ++){
                
                const int d = lane + output_slot * WARP_SIZE;

                if(d < head_dim){
                    float tile_output = 0.0f;

                    // 遍历当前tile内所有符合当前query要求的key, 因为他们都会被这个query影响。
                    for(int local_key = 0; local_key < query_valid_keys; local_key ++)
                        tile_output += scores[warp_id * TILE_K + local_key] * V_tile[local_key * head_dim + d];


                    // 代表当前lane这四个维度的sum,注意我们这里的所有量都是每个lane私有的，但我们通过更新保持同步。
                    // 一个维度上的所有scale后的value的和
                    output_accumulator[output_slot] = alpha * output_accumulator[output_slot] + tile_output;
                    
                }
            }




            
            // Step 8: Update online state.
            // We need to make sure every lane stores the same value;
            online_sum = alpha * online_sum + tile_sum;
            online_max = new_max;

        }



        // Before using next tile, every lane must finish all the work.
        __syncthreads();
    }










    // Now finish the tiling, we will have final normalization at here, 注意这里每个在每个warp中，相同位置的lane对应的维度相同
    // one warp writes one wuery row
    if(query_active && online_sum > 0.0f){
        
        #pragma unroll
        for(int output_slot = 0; output_slot < MAX_DIMS_PER_LANE; output_slot ++){

            const int d = lane + output_slot * WARP_SIZE;

            if(d < head_dim)
                output[query_row * head_dim + d] = output_accumulator[output_slot] / online;
        }
    }





}




// Host Wrapper
void cuda_flash_attention_multiquery_optimized(
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


    if(head_dim > 128)
        throw std::invalid_argument("Optimized tiled flash attention currently requires head_dim <= 128.");


    
    // The number of CTA's
    const int grid_size = (seq_len + QUERIES_PER_BLOCK - 1) / QUERIES_PER_BLOCK;





    // ========================================================
    // Dynamic shared memory
    //
    // K_tile:
    //
    //     TILE_K × head_dim
    //
    // V_tile:
    //
    //     TILE_K × head_dim
    //
    // Q_tile:
    //
    //     QUERIES_PER_BLOCK × head_dim
    //
    // scores:
    //
    //     QUERIES_PER_BLOCK × TILE_K
    // ========================================================
    const std::size_t shared_floats = static_cast<std::size_t>(
        TILE_K * head_dim + TILE_K * head_dim + QUERIES_PER_BLOCK * head_dim + QUERIES_PER_BLOCK * TILE_K
    );



    const std::size_t shared_bytes = shared_floats * sizeof(float);


    flash_attention_multiquery_optimized_kernel <<< grid_size, BLOCK_SIZE, shared_bytes, stream >>>(
        Q, K, V,
        output,
        seq_len, 
        head_dim
    );

}