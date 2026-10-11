#include "ops/cuda/matmul_operator.h"
#include "cuda/cuda_ops.cuh"

#include <stdexcept>

void CudaMatmulOperator::execute(
    const std::vector<const Tensor*>& inputs,
    const std::vector<Tensor*>& outputs
) const{
        // 1. Safety check and boundary validation.
    // Validate input/output
    if(inputs.size() != 2 || outputs.size() != 1)
        throw std::runtime_error("CpuMatmulOperator requires 2 inputs and 1 output.");

    const Tensor* A = inputs[0];
    const Tensor* B = inputs[1];

    Tensor* C = outputs[0];



    // null pointer safety check.
    if(A == nullptr || B == nullptr || C == nullptr)
        throw std::runtime_error("CpuMatmulOperator received a null tensor.");


    
    // Device Validation
    if(A -> device() != DeviceType::CPU ||
       B -> device() != DeviceType::CPU ||
       C -> device() != DeviceType::CPU)
        throw std::runtime_error("CpuMatmulOperator requires CPU tensors.");



    // Shape Validation
    if(A -> shape().size() != 2 ||
       B -> shape().size() != 2 || 
       C -> shape().size() != 2)
        throw std::runtime_error("CpuMatmulOperator requires 2D tensors.");

    


    // Exact dimensions and dimen check
    const int M = A -> shape()[0];
    const int K = A -> shape()[1];

    const int B_K = B -> shape()[0];
    const int N = B -> shape()[1];


    if(K != B_K)
        throw std::runtime_error("Matmul Input Shape mismatch.");
    
    if(C -> shape()[0] != M || C -> shape()[1] != N)
        throw std::runtime_error("Matmul Output Shape Mismatch.");


    cuda_matmul(A -> data(), B -> data(), C -> data(), M, K, N);
}