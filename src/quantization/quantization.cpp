#include "quantization/quantization.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>


QuantizationParams calculate_symmetric_quantization_params(
    const float* data,
    std::size_t size
){
    if(data == nullptr && size > 0)
        throw std::invalid_argument("Input data is null.");



    QuantizationParams params;


    // Empty Tensor
    if(size == 0){
        params.scale = 1.0f;
        params.zero_point = 0;

        return params;
    }




    float max_abs = 0.0f;

    for(std::size_t i = 0; i < size; i++)
        max_abs = std::max(max_abs, std::fabs(data[i]));


    

    /// Special case all values are 0
    if(max_abs == 0.0f){
        params.scale = 1.0f;

        params.zero_point = 0;

        return params;
    }




    // Now enter the normal case
    constexpr float INT8_SYMMETRIC_MAX = 127.0f;

    params.scale = max_abs / INT8_SYMMETRIC_MAX;

    params.zero_point = 0;


    return params;
}





// Quantize raw fp32 into raw int 8
void quantize_fp32_to_int8(
    const float* input,
    int8_t* output,
    std::size_t size,
    const QuantizationParams& params
){
    if(input == nullptr && size > 0)
        throw std::invalid_argument("Input data is null");

    if(output == nullptr && size > 0)
        throw std::invalid_argument("output data is null.");


    if(params.scale <= 0.0f)
        throw std::invalid_argument("Quantization scale must be positive");


    constexpr int32_t QUANT_MIN = -127;
    constexpr int32_t QUANT_MAX = 127;


    for(std::size_t i = 0; i < size; i++){
        const float scaled = input[i] / params.scale;

        int32_t quantized = static_cast<int32_t>(std::round(scaled));

        quantized += params.zero_point;


        quantized = std::clamp(quantized, QUANT_MIN, QUANT_MAX);


        output[i] = static_cast<int8_t>(quantized);
    }
}




// Helper function 
QuantizedTensor quantize_fp32_to_int8(
    const float* data,
    std::size_t size
){
    QuantizedTensor result;
    result.params = calculate_symmetric_quantization_params(data, size);

    result.data.resize(size);

    quantize_fp32_to_int8(
        data,
        result.data.data(),
        size,
        result.params
    );


    return result;
}





// dequantize x = scale * (q - zero_point)
void dequantize_int8_to_fp32(
    const int8_t* input,
    float* output,
    std::size_t size,
    const QuantizationParams& params
){
    if(input == nullptr && size > 0)
        throw std::invalid_argument("Input data is null.");


    if(output == nullptr && size > 0)
        throw std::invalid_argument("Output data is null.");


    for(std::size_t i = 0; i < size; i++){
        const int32_t quantized = static_cast<int32_t>(input[i]);

        output[i] = params.scale * static_cast<float>(quantized - params.zero_point);
    }

}




std::vector<float> dequantize_int8_to_fp32(
    const int8_t* data,
    std::size_t size,
    const QuantizationParams& params
){
    std::vector<float> result(size);

    dequantize_int8_to_fp32(data, result.data(), size, params);

    return result;
}