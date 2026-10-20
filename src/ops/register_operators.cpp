#include "ops/operator_registry.h"

#include "ops/cpu/gelu_operator.h"
#include "ops/cuda/gelu_operator.h"
#include "ops/cpu/matmul_operator.h"
#include "ops/cuda/matmul_operator.h"
#include "ops/cpu/layernorm_operator.h"
#include "ops/cuda/layernorm_operator.h"
#include "ops/cpu/attention_scores_operator.h"
#include "ops/cuda/attention_scores_operator.h"
#include "ops/cpu/attention_softmax_operator.h"
#include "ops/cuda/attention_softmax_operator.h"


#include <memory>  // C++ 内存管理工具，尤其是指针


void OperatorRegistry::register_builtin_operators(){

    // CPU at here
    

 

    registry_operator(
        "LayerNorm",
        DeviceType::CPU,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CpuLayerNormOperator>();
        }
    );


    registry_operator(
        "AttentionScores",
        DeviceType::CPU,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CpuAttentionScoresOperator>();
        }
    );


    registry_operator(
        "AttentionSoftmax",
        DeviceType::CPU,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CpuAttentionSoftmaxOperator>();
        }
    );


    // CUDA at here
   


     
        

    registry_operator(
        "LayerNorm",
        DeviceType::CUDA,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CudaLayerNormOperator>();
        }
    );


    registry_operator(
        "AttentionScores",
        DeviceType::CUDA,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CudaAttentionScoresOperator>();
        }
    );


    registry_operator(
        "AttentionSoftmax",
        DeviceType::CUDA,
        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CudaAttentionSoftmaxOperator>();
        }
    );

}



