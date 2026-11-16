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





}