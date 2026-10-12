#include "ops/cuda/layernorm_operator.h"
#include "cuda/cuda_ops.cuh"

#include <stdexcept>


void CudaLayerNormOperator::execute(
    const std::vector<const Tensor*>& inputs,
    const std::vector<Tensor*>& outputs
) const
{
    if(inputs.size() != 3 || outputs.size() != 1)
        throw std::runtime_error("CpuLayerNormOperator expects 3 inputs and 1 output.");

    
    const Tensor* input = inputs[0];
    const Tensor* gamma = inputs[1];
    const Tensor* beta = inputs[2];

    Tensor* output = outputs[0];




    if(input == nullptr || gamma == nullptr || beta == nullptr || output == nullptr)
        throw std::runtime_error("CpuLayerNormOperator received a null tensor.");


    if(input->device() != DeviceType::CUDA || gamma->device() != DeviceType::CUDA || beta->device() != DeviceType::CUDA || output->device() != DeviceType::CUDA)
        throw std::runtime_error("CpuLayerNormOperator requires CPU tensors.");

    if(input->size() != gamma->size() || input->size() != beta->size() || input->size() != output->size())
        throw std::runtime_error("Layernorm Size Mismatch.");

    constexpr float epsilon = 1e-5f;



    cuda_layer_norm(input->data(), gamma->data(), beta->data(), output->data(), static_cast<int>(input->size()), epsilon);

}