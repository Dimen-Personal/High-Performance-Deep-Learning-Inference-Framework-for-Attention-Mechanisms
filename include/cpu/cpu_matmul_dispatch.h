#pragma once


enum class CpuMatmulMode{
    FP32,
    INT8
};



struct CpuMatmulConfig{
    CpuMatmulMode mode = CpuMatmulMode::FP32;
    int thread_count = 1;
};



// Unified CPU MatMul Dispatch
//
// Input:
//     FP32 A
//     FP32 B
//
// Output:
//     FP32 C
//
// FP32 mode:
//     FP32 OpenMP MatMul
//
// INT8 mode:
//     FP32
//       ↓
//     INT8 quantization
//       ↓
//     INT8 OpenMP MatMul
//       ↓
//     FP32 output
void cpu_matmul_dispatch(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N,
    const CpuMatmulConfig& config
);