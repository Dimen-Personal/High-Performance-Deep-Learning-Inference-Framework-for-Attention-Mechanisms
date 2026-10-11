#ifndef CPU_MATMUL_OPERATOR_H
#define CPU_MATMUL_OPERATOR_H

#include "ops/cpu_operator.h"

class CpuMatmulOperator final : public CpuOperator{
    public:
        
        void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs
        ) const override;
};


#endif