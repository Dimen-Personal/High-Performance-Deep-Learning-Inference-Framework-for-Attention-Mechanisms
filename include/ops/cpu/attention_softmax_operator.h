#ifndef CPU_ATTENTION_SOFTMAX_OPERATOR_H
#define CPU_ATTENTION_SOFTMAX_OPERATOR_H

#include "ops/cpu_operator.h"

class CpuAttentionSoftmaxOperator final : public CpuOperator{

    public:
        void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs
        )const override;
};


#endif