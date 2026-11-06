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
    const int query_row = blockIdx.x;
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





}