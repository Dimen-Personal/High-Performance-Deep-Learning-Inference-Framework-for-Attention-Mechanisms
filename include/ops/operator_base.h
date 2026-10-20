#ifndef OPERATOR_BASE_H
#define OPERATOR_BASE_H

#include "tensor/tensor.h"
#include "runtime/execution_context.h"
#include <vector>

class Operator{
    
    public:
        virtual ~Operator() = default;  

        virtual void execute(
            const std::vector<const Tensor*>& inputs,
            const std::vector<Tensor*>& outputs,
            const ExecutionContext& context
        )const = 0;    // = 0 means pure virtual fucntion: 强制具体的派生类提供自己的实现，除非它也打算为抽象类
};

#endif