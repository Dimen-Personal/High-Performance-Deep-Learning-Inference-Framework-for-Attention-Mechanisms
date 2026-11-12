#pragma once





// 注意我们这里的 bias_add 是在所有token的同一个维度加上一个 bias
void bias_add(
    const float* input,
    const float* bias,
    float* output,
    int M,
    int N
);





void unfused_matmul_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* temp_matmul,
    float* temp_bias,
    float* output,
    int M,
    int K, 
    int N
);




void fused_matmul_bias_gelu(
    const float* A,
    const float* B,
    const float* bias,
    float* output,
    int M,
    int K, 
    int N
);






