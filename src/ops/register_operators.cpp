#include "ops/operator_registry.h"

#include "ops/cpu/gelu_operator.h"
#include "ops/cuda/gelu_operator.h"
#include "ops/cpu/matmul_operator.h"
#include "ops/cuda/matmul_operator.h"


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


    registry_operator(
        "MatMul",
        DeviceType::CPU,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CpuMatmulOperator>();
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


    registry_operator(
        "Matmul",
        DeviceType::CUDA,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CudaMatmulOperator>();
        }
    );
        

}



