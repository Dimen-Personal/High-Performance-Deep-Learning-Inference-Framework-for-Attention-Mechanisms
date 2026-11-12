#include "cpu/fused_matmul_bias_gelu.h"
#include "ops/operator.h"

#include <cmath>
#include <stdexcept>




// Notive input is [M, N];
void bias_add(
    const float* input,
    const float* bias,
    float* output,
    int M,
    int N
){
    // Validation
    if(input == nullptr)
        throw std::invalid_argument("bias_add input is null.");

    if(bias == nullptr)
        throw std::invalid_argument("bias_add bias is null.");

    if(output == nullptr)
        throw std::invalid_argument("bias_add output is null.");


    if(M <= 0 || N <= 0)
        throw std::invalid_argument("bias_ad dimensions must be positive.");




    // Main
    for(int i = 0; i < M; i++)
        for(int j = 0; j < N; j++)
            output[i * N + j] = input[i * N + j] + bias[j];
}







void unfused_matmul_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* temp_matmul,
    float* temp_bias,
    float* output,
    int M,
    int K, 
    int N
){
    // Validation
    if(A == nullptr || B == nullptr || bias == nullptr ||
       temp_matmul == nullptr || temp_bias == nullptr || output == nullptr )
       throw std::invalid_argument("unfused matmul bias geu received null pointer");

    
    if(M <= 0 || K <= 0 || N <= 0)
        throw std::invalid_argument("unfused matmul bias felu domensions must be positive.");



    matmul(A, B, temp_matmul, M, K, N);

    bias_add(temp_matmul, bias, temp_bias, M, N);

    gelu(temp_bias, output, M * N);
}






// A[M, K]
// B[K, N] weight 1
// bias[N]
// output[M, N]
// Only touch global memory at output
void fused_matmul_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* output,
    int M,
    int K, 
    int N
){
    // Validation
    if(A == nullptr || B == nullptr || bias == nullptr || output == nullptr )
       throw std::invalid_argument("fused matmul bias geu received null pointer");

    
    if(M <= 0 || K <= 0 || N <= 0)
        throw std::invalid_argument("fused matmul bias felu domensions must be positive.");


    
    const float inv_sqrt_2 = 1.0f / std::sqrt(2.0f);




    // Fused Computation
    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++){
            float sum = 0.0f;

            for(int k = 0; k < K; k++)
                sum += A[i * K + k] * B[k * N + j];

            // matmul[i][j] finish

            // bias add
            const float biased = sum + bias[j];

            
            // GELU
            const float activated = 0.5f * biased * (1.0f + std::erf(biased * inv_sqrt_2));


            output[i * N + j] = activated;
        }
    }



}



