#include <cuda_runtime.h>


__global__
void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row >= M || col >= N)
        return;

    float sum = 0.0f;


    for(int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}



void cuda_matmul(const float* A, const float* B, float* C, int M, int K, int N, cudaStream_t stream){
    dim3 block(16,16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);


    matmul_kernel <<< grid, block, 0, stream >>>(A, B, C, M, K, N);

}