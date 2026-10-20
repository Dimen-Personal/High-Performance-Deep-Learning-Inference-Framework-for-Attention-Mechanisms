#ifndef CPU_GELU_OPERATOR_H
#define CPU_GELU_OPERATOR_H

#include "ops/cpu_operator.h"

class CpuGeluOperator final : public CpuOperator{   // final means this class can't be inherented by other classes.
    public:
        void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs,
            const ExecutionContext&    
        ) const override;   // Override 保证函数签名和父类一致，进行double check. 
};

#endif