#pragma once

void fp32_matmul_openmp(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K, 
    int N,
    int thread_count
);


