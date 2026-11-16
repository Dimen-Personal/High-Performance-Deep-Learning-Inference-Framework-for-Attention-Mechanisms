#pragma once



void cache_aware_attention_tiled(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int seq_len,
    int head_dim,
    int query_block_size = 16,
    int key_block_size = 32
);