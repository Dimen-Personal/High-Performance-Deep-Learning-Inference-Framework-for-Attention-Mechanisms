#include "ops/cpu/gelu_operator.h"
#include "ops/operator.h"
#include "ops/operator_registrar.h"

#include <stdexcept>
#include <memory>


void CpuGeluOperator::execute(
    const std::vector <const Tensor*>& inputs,
    const std::vector <Tensor*>& outputs
) const
{


    // Safety Check
    if(inputs.size() != 1 || outputs.size() != 1)
        throw std::runtime_error("CpuGeluOperator expects 1 inputs and outputs");

    const Tensor* input = inputs[0];
    Tensor* output = outputs[0];

    if(input == nullptr || output == nullptr)
        throw std::runtime_error("CpuGeluOperator received a null tensor");

    if(input->shape() != output -> shape())
        throw std::runtime_error("Gelu input/output shape mismatch.");

    if(input->device() != DeviceType::CPU || output->device() != DeviceType::CPU)
        throw std::runtime_error("CpuGeluOperator requires CPU Tensors.");

    gelu(input->data(), output->data(), static_cast<int>(input->size()));

}


// Only could be seen in this file.
namespace{
    OperatorRegistrar cpu_gelu_registrar(
        "GELU",
        DeviceType::CPU,

        []() -> std::unique_ptr<Operator>{
            return std::make_unique<CpuGeluOperator>();
        }
    );
}
