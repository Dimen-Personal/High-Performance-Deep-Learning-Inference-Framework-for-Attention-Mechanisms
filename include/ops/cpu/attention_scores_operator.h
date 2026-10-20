#ifndef CPU_ATTENTION_SCORES_OPERATOR_H
#define CPU_ATTENTION_SCORES_OPERATOR_H

#include "ops/cpu_operator.h"

class CpuAttentionScoresOperator final : public CpuOperator{

    public:
        void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs,
            const ExecutionContext&    // Means [for satisfying interface] but not use
        )const override;
};


#endif