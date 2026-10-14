#include "cuda/cuda_ops.cuh"
#include "ops/operator.h"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>


bool comapre_results(









int main(){

    // We will do three reference at here.
    const int seq_len = 4;
    const float tolerance = 1e-5f;
    const int blocks_per_row = 4;

    std::vector<float> scores = {
        2.0f,     -INFINITY, -INFINITY, -INFINITY,

        1.0f,      3.0f,     -INFINITY, -INFINITY,

        0.5f,      1.0f,      2.0f,     -INFINITY,

        1.0f,      2.0f,      3.0f,      4.0f
    };


    std::vector<float> cpu_weights(seq_len * seq_len);
    std::vector<float> single_weights(seq_len * seq_len);
    std::vector<float> multi_weights(seq_len * seq_len);


    // CPU Reference
    attention_softmax(scores.data(), cpu_weights.data(), seq_len);



    // CUDA Allocation
    float* d_scores = nullptr;
    
    float* d_single_weights = nullptr;
    float* d_multi_weights = nullptr;
    
    float* d_partial_max = nullptr;
    float* d_partial_sum = nullptr;

    float* d_row_max = nullptr;
    float* d_row_sum = nullptr;




    //const std::size_t bytes = static_cast<std::size_t>(seq_len * seq_len) * sizeof(float);
    const std::size_t matrix_bytes = static_cast<std::size_t>(seq_len * seq_len) * sizeof(float);
    const std::size_t partial_bytes = static_cast<std::size_t>(seq_len * blocks_per_row) * sizeof(float);
    const std::size_t row_bytes = static_cast<std::size_t>(seq_len) * sizeof(float);

    cudaMalloc(&d_scores, matrix_bytes);
    cudaMalloc(&d_single_weights, matrix_bytes);
    cudaMalloc(&d_multi_weights, matrix_bytes);
    cudaMalloc(&d_partial_max, partial_bytes);
    cudaMalloc(&d_partial_sum, partial_bytes);
    cudaMalloc(&d_row_max, row_bytes);
    cudaMalloc(&d_row_sum, row_bytes);

    cudaMemcpy(d_scores, scores.data(), matrix_bytes, cudaMemcpyHostToDevice);





    // CUDA Attention softmax (single block version)
    cuda_attention_softmax(d_scores, d_single_weights, seq_len);

    cudaDeviceSynchronize();

    cudaMemcpy(single_weights.data(), d_single_weights, matrix_bytes, cudaMemcpyDeviceToHost);



    // Comapare
    bool correct = true;

    for(int i = 0; i < seq_len * seq_len; i++){
        const float cpu = cpu_weights[i];
        const float gpu = gpu_weights[i];

        if(std::isinf(cpu) && std::isinf(gpu))
            continue;


        const float diff = std::fabs(cpu - gpu);

        if(diff > tolerance){
            correct = false;

            std::cout << "Mismatch at index" << i << std::endl;
            std::cout << "CPU = " << cpu << std::endl;
            std::cout << "GPU = " << gpu << std::endl;
            std::cout << "Diff = " << diff << std::endl;

            break;
        }
    }



    std::cout << "CUDA Attention Softmax: " << (correct ? "PASS" : "FAIL") << std::endl;


    cudaFree(d_scores);
    cudaFree(d_weights);
    

    return correct ? 0 : 1;

}