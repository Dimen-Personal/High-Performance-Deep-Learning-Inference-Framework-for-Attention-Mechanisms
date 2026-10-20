#include "ops/cuda/gelu_operator.h"
#include "cuda/cuda_ops.cuh"
#include "ops/operator_registrar.h"

#include <stdexcept>


void CudaGeluOperator::execute(
    const std::vector<const Tensor*>& inputs,
    const std::vector<Tensor*>& outputs,
     const ExecutionContext& context
)const
{
    if(inputs.size() != 1 || outputs.size() != 1)
        throw std::runtime_error("CudaGeluOperator expects 1 inputs and 1 outputs");

    const Tensor* input = inputs[0];
    Tensor* output = outputs[0];

    if(input == nullptr || output == nullptr)
        throw std::runtime_error("CudaGeluOperator received a null tensor");

    if(input->shape() != output -> shape())
        throw std::runtime_error("Gelu input/output shape mismatch.");

    if(input->device() != DeviceType::CUDA || output->device() != DeviceType::CUDA)
        throw std::runtime_error("CudaGeluOperator requires Cuda Tensors.");



    cuda_gelu(input->data(), output->data(), static_cast<int>(input -> size()), static_cast<cudaStream_t>(context.stream));
}



namespace{
    OperatorRegistrar cuda_gelu_registrar(
        "GELU",
        DeviceType::CUDA,

        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CudaGeluOperator>();
        }
    );
}