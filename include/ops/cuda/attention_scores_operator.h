#ifndef CUDA_ATTENTION_SCORES_OPERATOR_H
#define CUDA_ATTENTION_SCORES_OPERATOR_H

#include "ops/cuda_operator.h"

class CudaAttentionScoresOperator final : public CudaOperator{

    public:
        void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs
        )const override;
};


#endif