#include "quantization/int8_matmul.h"

#include <cstdint>
#include <stdexcept>




// Helper
static void validate_int8_matmul_inputs(
    const int8_t* A,
    const int8_t* B,
    float* C,
    int M,
    int K,
    int N,
    const QuantizationParams& params_A,
    const QuantizationParams& params_B
){
    if(A == nullptr)
        throw std::invalid_argument("INT8 Matmul A is null.");


    if(B == nullptr)
        throw std::invalid_argument("INT8 Matmul B is null.");


    if(C == nullptr)
        throw std::invalid_argument("INT8 Matmul C is null.");

    
    if(M <= 0 || K <= 0 || N <= 0)
        throw std::invalid_argument("INT8 MatMul dimensions must be positive.");


    if(params_A.scale <= 0.0f || params_B.scale <= 0.0f)
        throw std::invalid_argument("INT8 MatMul quantization scale must be positive.");

}








// single thread
void int8_matmul(
    const int8_t* A,
    const int8_t* B,
    float* C,
    int M,
    int K,
    int N,
    const QuantizationParams& params_A,
    const QuantizationParams& params_B
){
    
    validate_int8_matmul_inputs(A, B, C, M, K, N, params_A, params_B);




    const float output_scale = params_A.scale * params_B.scale;

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++){
            int32_t accumulator = 0;

            for(int k = 0; k < K; k++){
                const int32_t a = static_cast<int32_t>(A[i * K + k]) - params_A.zero_point;
                const int32_t b = static_cast<int32_t>(B[k * N + j]) - params_B.zero_point;

                accumulator += a * b;
            }


            C[i * N + j] = static_cast<float>(accumulator) * output_scale;
        }
    

}








void int8_matmul_openmp(
    const int8_t* A,
    const int8_t* B,
    float* C,
    int M,
    int K,
    int N,
    const QuantizationParams& params_A,
    const QuantizationParams& params_B,
    int thread_count
){
  
    validate_int8_matmul_inputs(A, B, C, M, K, N, params_A, params_B);


    if(thread_count < 0)
        throw std::invalid_argument("Thread count must be positive");






    const float output_scale = params_A.scale * params_B.scale;



    // Row-Level OpenMP parallelization. Each thread owns seperate output rows.
    // 用给予的threads数量拆开下面的i，给每个thread在运行前就分配好要运行的行数。
    #pragma omp parallel for \
        num_threads(thread_count) \ 
        schedule(static)

    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++){
            int32_t accumulator = 0;

            for(int k = 0; k < K; k++){
                const int32_t a = static_cast<int32_t>(A[i * K + k]) - params_A.zero_point;
                const int32_t b = static_cast<int32_t>(B[k * N + j]) - params_B.zero_point;

                accumulator += a * b;
            }


            C[i * N + j] = static_cast<float>(accumulator) * output_scale;
        }
    

}





