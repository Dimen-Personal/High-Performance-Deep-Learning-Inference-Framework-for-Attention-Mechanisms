#ifndef CPU_LAYERNORM_OPERATOR_H
#define CPU_LAYERNORM_OPERATOR_H

#include "ops/cpu_operator.h"

class CpuLayerNormOperator final : public CpuOperator{

    public:
        void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs
        )const override;
};


#endif