#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// On current int 8 quantization, we will keep zero point as 0 and range as -127 to 127.


struct QuantizationParams{
    float scale = 1.0f;

    int32_t zero_point = 0;
};


struct QuantizedTensor{
    std::vector<int8_t> data;

    QuantizationParams params;
};



// calculate scale based on the input fp32
QuantizationParams calculate_symmetric_quantization_params(
    const float* data,
    std::size_t size
);



QuantizedTensor quantize_fp32_to_int8(
    const float* data,
    std::size_t size
);



std::vector<float> dequantize_int8_to_fp32(
    const int8_t* data,
    std::size_t size,
    const QuantizationParams& params
);




// Raw buffer version, for future use
void quantize_fp32_to_int8(
    const float* input,
    int8_t* output,
    std::size_t size,
    const QuantizationParams& params
);

void dequantize_int8_to_fp32(
    const int8_t* input,
    float* output,
    std::size_t size,
    const QuantizationParams& params
);





