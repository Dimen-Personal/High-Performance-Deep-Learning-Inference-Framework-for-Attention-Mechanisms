#ifndef CUDA_ATTENTION_SOFTMAX_OPERATOR_H
#define CUDA_ATTENTION_SOFTMAX_OPERATOR_H

#include "ops/cuda_operator.h"

class CudaAttentionSoftmaxOperator final : public CudaOperator{

    public:
        void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs
        )const override;
};


#endif