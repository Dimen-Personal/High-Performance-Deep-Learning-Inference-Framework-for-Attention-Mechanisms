#include "cpu/fp32_matmul.h"

#include <stdexcept>
#include <omp.h>


void fp32_matmul_openmp(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
    int thread_count
){

    if(A == nullptr)
        throw std::invalid_argument("FP32 Matmul A is null.");


    if(B == nullptr)
        throw std::invalid_argument("FP32 Matmul B is null.");


    if(C == nullptr)
        throw std::invalid_argument("FP32 Matmul C is null.");

    
    if(M <= 0 || K <= 0 || N <= 0)
        throw std::invalid_argument("FP32 MatMul dimensions must be positive.");

 
    if(thread_count < 0)
        throw std::invalid_argument("Thread count must be positive");





    // Row-Level OpenMP parallelization. Each thread owns seperate output rows.
    // 用给予的threads数量拆开下面的i，给每个thread在运行前就分配好要运行的行数。
    #pragma omp parallel for \
        num_threads(thread_count) \ 
        schedule(static)

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++){
            float sum = 0;

            for(int k = 0; k < K; k++){
                const float a = A[i * K + k];
                const float b = B[k * N + j];

                sum += a * b;
            }


            C[i * N + j] = sum;
        }
    
}