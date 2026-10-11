#include "ops/operator_registry.h"

#include "ops/cpu/gelu_operator.h"
#include "ops/cuda/gelu_operator.h"

#include <memory>  // C++ 内存管理工具，尤其是指针


void OperatorRegistry::register_builtin_operators(){

    // CPU at here
    registry_operator(
        "GELU",
        DeviceType::CPU,
        []() -> std::unique_ptr<Operator>{      //[capture](parameters) -> return_type {function body}   Lambada
            return std::make_unique<CpuGeluOperator>();
        }
    );


     




    // CUDA at here
    registry_operator(
        "GELU",
        DeviceType::CUDA,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CudaGeluOperator>();
        }
    );
        

}



