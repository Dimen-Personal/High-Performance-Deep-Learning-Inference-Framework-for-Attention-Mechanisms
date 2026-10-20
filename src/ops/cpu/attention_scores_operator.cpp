#include "ops/cpu/attention_scores_operator.h"
#include "ops/operator.h"
#include "ops/operator_registrar.h"

#include <stdexcept>


void CpuAttentionScoresOperator::execute(
    const std::vector<const Tensor*>& inputs,
    const std::vector<Tensor*>& outputs,
    const ExecutionContext&    
) const
{
    if(inputs.size() != 2 || outputs.size() != 1)
        throw std::runtime_error("CpuAttentionScoresOperator expects 2 inputs and 1 output.");



    const Tensor* Q = inputs[0];
    const Tensor* K = inputs[1];

    Tensor* scores = outputs[0];


     
    if(Q == nullptr || K == nullptr || scores == nullptr)
        throw std::runtime_error("CpuAttentionScoresOperator received a null tensor.");


    
    if(
        Q->device() != DeviceType::CPU ||
        K->device() != DeviceType::CPU ||
        scores->device() != DeviceType::CPU
    )
        throw std::runtime_error("CpuAttentionScoresOperator requires CPU tensors.");


    
    if(
        Q->shape().size() != 2 ||
        K->shape().size() != 2 ||
        scores->shape().size() != 2
    )
        throw std::runtime_error("AttentionScores requires 2D tensors.");


     

    const int seq_len = Q->shape()[0];
    const int head_dim = Q->shape()[1];


   
    if(Q->shape() != K->shape())
        throw std::runtime_error("AttentionScores Q/K shape mismatch.");


    
    if(
        scores->shape()[0] != seq_len ||
        scores->shape()[1] != seq_len
    )
        throw std::runtime_error("AttentionScores output shape mismatch.");


    
    attention_scores(
        Q->data(),
        K->data(),
        scores->data(),
        seq_len,
        head_dim
    );
}


namespace{
    OperatorRegistrar cpu_gelu_registrar(
        "AttentionScores",
        DeviceType::CPU,

        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CpuAttentionScoresOperator>();
        }
    );
}
