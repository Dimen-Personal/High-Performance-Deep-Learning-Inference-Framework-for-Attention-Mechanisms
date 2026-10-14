#include "cuda/cuda_ops.cuh"
#include "ops/operator.h"

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <vector>


bool comapre_results(
    const std::vector<float>& reference,
    const std::vector<float>& result,
    float tolerance,
    const char* name
){
    if(reference.size() != result.size())
        return false;

    for(std::size_t i = 0; i < reference.size(); i++){
        const float ref = reference[i];
        const float value = result[i];

        if(std::isnan(ref) || std::isnan(value)){
            std::cout << name << ": NaN at index " << i << std::endl;
            return false;
        }


        const float diff = std::fabs(ref - value);
        if(diff > tolerance){
            std::cout << name << ": mismatch at index " << i << std::endl;
            std::cout << "Reference = " << ref << std::endl;
            std::cout << "Result = " << value << std::endl;
            std::cout << "Diff = " << diff << std::endl;

            return false;
        }
    }


    return true;
}



bool check_row_sums(
    const std::vector<float>& weights,
    int seq_len,
    float tolerance,
    const char* name
){
    for(int row = 0; row < seq_len; row++){
        float sum = 0.0f;

        for(int col = 0; col < seq_len; col ++)
            sum += weights[row * seq_len + col];

        if(std::fabs(sum - 1.0f) > tolerance){
            std::cout << name << ": row" << row << " sum = " << sum <<std::endl;

            return false;
        }
    }


    return true;
}







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




    // CUDA Attention softmax (multi block version)
    cuda_attention_softmax_multiblock(
        d_scores,
        d_multi_weights,

        d_partial_max,
        d_partial_sum,

        d_row_max,
        d_row_sum,

        seq_len,
        blocks_per_row
    );

    cudaDeviceSynchronize();

    cudaMemcpy(multi_weights.data(), d_multi_weights, matrix_bytes, cudaMemcpyDeviceToHost);





    // Verification and correctness.
    const bool single_correct = comapre_results(
        cpu_weights,
        single_weights,
        tolerance,
        "single block"
    );

    const bool multi_correct = comapre_results(
        cpu_weights,
        multi_weights,
        tolerance,
        "multi block"
    );

    const bool single_sum_correct = check_row_sums(
        single_weights,
        seq_len,
        tolerance, 
        "Single block"
    );

    const bool multi_sum_correct = check_row_sums(
        multi_weights,
        seq_len,
        tolerance,
        "Multi block"
    );


    std::cout << "single block softmax: " << (single_correct && single_sum_correct ? "PASS" : "FAIL") << std::endl;
    std::cout << "multi block softmax: " << (multi_correct && multi_sum_correct ? "PASS" : "FAIL") << std::endl;


    cudaFree(d_scores);
    cudaFree(d_single_weights);
    cudaFree(d_multi_weights);
    cudaFree(d_partial_max);
    cudaFree(d_partial_sum);
    cudaFree(d_row_max);
    cudaFree(d_row_sum);


    const bool correct = single_correct && multi_correct && single_sum_correct && multi_sum_correct;

    return correct ? 0 : 1;

    /*
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
*/


}