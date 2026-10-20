#ifndef CUDA_LAYERNORM_OPERATOR_H
#define CUDA_LAYERNORM_OPERATOR_H

#include "ops/cuda_operator.h"

class CudaLayerNormOperator final : public CudaOperator{

    public:
        void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs,
             const ExecutionContext& context
        )const override;
};


#endif