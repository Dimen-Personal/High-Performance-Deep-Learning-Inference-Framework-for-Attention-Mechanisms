#include "cpu/cache_aware_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>




void cache_aware_attention_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    int query_block_size,
    int key_block_size
){
    if(Q == nullptr || K == nullptr || V == nullptr || output == nullptr)
        throw std::invalid_argument("cache aware attention tiled received null pointer.");

    if(seq_len <= 0 || head_dim <= 0 || query_block_size <= 0 || key_block_size <= 0)
        throw std::invalid_argument("cache aware attention tilled dimension must be positive.");


    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));





    std::vector<float> q_tile(static_cast<std::size_t>(query_block_size) * head_dim);
    std::vector<float> k_tile(static_cast<std::size_t>(key_block_size) * head_dim);
    std::vector<float> v_tile(static_cast<std::size_t>(key_block_size) * head_dim);
    std::vector<float> score_tile(static_cast<std::size_t>(query_block_size) * key_block_size);





    // Online softmax state for one query block
    std::vector<float> row_max(query_block_size);
    std::vector<float> row_sum(query_block_size);
    std::vector<float> output_accumulator(static_cast<std::size_t>(query_block_size) * head_dim);






    // Query Blocking
    for(int query_start = 0; query_start < seq_len; query_start += query_block_size){

        const int query_count = std::min(query_block_size, seq_len - query_start);


        // Loading query tile
        for(int q = 0; q < query_count; q++){
            const int global_query = query_start + q;

            for(int d = 0; d < head_dim; d++)
                q_tile[q * head_dim + d] = Q[global_query * head_dim + d];

        }



        for(int q = 0; q < query_count; q++){
            row_max[q] = -std::numeric_limits<float>::infinity();
            row_sum[q] = 0.0f;

            // 这里最后 output_accumulator还是要换回去的
            for(int d = 0; d < head_dim; d++)
                output_accumulator[q * head_dim + d] = 0.0f;

        }
    
    

        // casual mask.
        const int maximum_key  = query_start + query_count - 1;




        // K/V Blocking
        for(int key_start = 0; key_start <= maximum_key; key_start += key_block_size){
            
            const int key_count = std::min(key_block_size, maximum_key - key_start + 1);

            // Load KV tile
            for(int k = 0; k < key_count; k++){
                const int global_key = key_start + k;

                for(int d = 0; d < head_dim; d++){
                    k_tile[k * head_dim + d] = K[global_key * head_dim + d];

                    v_tile[k * head_dim + d] = V[global_key * head_dim + d];
                }
            }



            // Compute score tile
            // [query_count, key_count]
            for(int q = 0; q < query_count; q++){
                const int global_query = query_start + q;

                for(int k = 0 ; k < key_count; k++){
                    const int global_key = key_start + k;

                    if(global_key > global_query){
                        score_tile[q * key_block_size + k] = -std::numeric_limits<float>::infinity();

                        continue;
                    }


                    // Q dot K
                    float score = 0.0f;

                    for(int d = 0; d < head_dim; d++)
                        score += q_tile[q * head_dim + d] * k_tile[k * head_dim + d];
                    
                    score_tile[q * key_block_size + k] = score * scale;
                }
            }





            // online softmax
            // 对这一组里的每一个query去做
            for(int q = 0; q < query_count; q++){
                const int global_query = query_start + q;

                const int valid_key_count = std::min(key_count, global_query - key_start + 1);


                if(valid_key_count <= 0)
                    continue;



                // Tile Maximum
                float tile_max = -std::numeric_limits<float>::infinity();


                for(int k = 0; k < valid_key_count; k++)
                    tile_max = std::max(tile_max, score_tile[q * key_block_size + k]);



                

                // Online maximum
                const float old_max = row_max[q];
                const float new_max = std::max(old_max, tile_max);





                // Rescale old softmax state
                const float alpha = (row_sum[q] == 0.0f) ? 0.0f : std::exp(old_max - new_max);

                for(int d = 0; d < head_dim; d++)
                    output_accumulator[q * head_dim + d] *= alpha;



                
                
                float tile_sum = 0.0f;

                for(int k = 0; k < valid_key_count; k++){
                    const float beta = std::exp(score_tile[q * key_block_size + k] - new_max);

                    tile_sum += beta;


                    // 后面加上的就是现在这个tile的weight * Value
                    for(int d = 0; d < head_dim; d++)
                        output_accumulator[q * head_dim + d] += beta * v_tile[k * head_dim + d];



                }



                // update row sum and row max for final normalization
                row_sum[q] = alpha * row_sum[q] + tile_sum;

                row_max[q] = new_max; 
            }

        }
    

        
        
        
        // Final normalization
        for(int q = 0; q < query_count; q++){
            const int global_query = query_start + q;

            for(int d = 0; d < head_dim; d++)
                output[global_query * head_dim + d] = output_accumulator[q * head_dim + d] / row_sum[q];
        }
    
    
    
    }


}