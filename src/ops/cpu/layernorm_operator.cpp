#include "ops/cpu/layernorm_operator.h"
#include "ops/operator.h"
#include "ops/operator_registrar.h"

#include <stdexcept>


void CpuLayerNormOperator::execute(
    const std::vector<const Tensor*>& inputs,
    const std::vector<Tensor*>& outputs
) const
{
    if(inputs.size() != 3 || outputs.size() != 1)
        throw std::runtime_error("CpuLayerNormOperator expects 3 inputs and 1 output.");

    
    const Tensor* input = inputs[0];
    const Tensor* gamma = inputs[1];
    const Tensor* beta = inputs[2];

    Tensor* output = outputs[0];




    if(input == nullptr || gamma == nullptr || beta == nullptr || output == nullptr)
        throw std::runtime_error("CpuLayerNormOperator received a null tensor.");


    if(input->device() != DeviceType::CPU || gamma->device() != DeviceType::CPU || beta->device() != DeviceType::CPU || output->device() != DeviceType::CPU)
        throw std::runtime_error("CpuLayerNormOperator requires CPU tensors.");

    if(input->size() != gamma->size() || input->size() != beta->size() || input->size() != output->size())
        throw std::runtime_error("Layernorm Size Mismatch.");

    constexpr float epsilon = 1e-5f;



    layer_norm(input->data(), gamma->data(), beta->data(), output->data(), static_cast<int>(input->size()), epsilon);

}


namespace{
    OperatorRegistrar cpu_layernorm_registrar(
        "LayerNorm",
        DeviceType::CPU,

        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CpuLayerNormOperator>();
        }
    );
}