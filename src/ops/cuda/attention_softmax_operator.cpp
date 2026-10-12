#include "ops/cuda/attention_softmax_operator.h"
#include "cuda/cuda_ops.cuh"

#include <stdexcept>


void CudaAttentionSoftmaxOperator::execute(
    const std::vector<const Tensor*>& inputs,
    const std::vector<Tensor*>& outputs
) const
{
 
    if(inputs.size() != 1 || outputs.size() != 1)
        throw std::runtime_error("CpuAttentionSoftmaxOperator expects 1 input and 1 output.");


     
    const Tensor* input = inputs[0];
    Tensor* output = outputs[0];


    
    if(input == nullptr || output == nullptr)
        throw std::runtime_error("CpuAttentionSoftmaxOperator received a null tensor.");


  
    if(
        input->device() != DeviceType::CUDA ||
        output->device() != DeviceType::CUDA
    )
        throw std::runtime_error("CpuAttentionSoftmaxOperator requires CPU tensors.");



  
    if(input->shape() != output->shape())
        throw std::runtime_error("AttentionSoftmax input/output shape mismatch.");


     
    if(input->shape().size() != 2)
        throw std::runtime_error("AttentionSoftmax requires 2D tensors.");


    
    if(input->shape()[0] != input->shape()[1])throw std::runtime_error("AttentionSoftmax requires a square matrix.");


     
    const int seq_len = input->shape()[0];


   
    cuda_attention_softmax(
        input->data(),
        output->data(),
        seq_len
    );
}