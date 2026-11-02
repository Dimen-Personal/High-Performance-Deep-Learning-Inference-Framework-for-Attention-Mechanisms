#include "quantization/int8_matmul.h"

#include <cstdint>
#include <stdexcept>

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