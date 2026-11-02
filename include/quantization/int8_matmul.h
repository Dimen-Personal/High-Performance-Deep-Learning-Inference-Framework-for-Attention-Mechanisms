#pragma once

#include "quantization/quantization.h"
#include <cstdint>


/*
A: M * K
B: K * N
C: M * N
*/
void int8_matmul(
    const int8_t* A,
    const int8_t* B,
    float* C,
    int M,
    int K,
    int N,
    const QuantizationParams& params_A,
    const QuantizationParams& params_B
);