#include "cuda/cuda_ops.cuh"

#include "graph/graph.h"
#include "graph/node.h"

#include "ops/operator.h"

#include "tensor/tensor.h"

#include "runtime/cuda_memory_pool.h"
#include "runtime/device_transfer.h"
#include "runtime/executor.h"


#include <cmath>
#include <iostream>
#include <vector>


int main(){
    const int size = 513;
    const float tolerance = 1e-5f;

    const std::vector<int> shape = {size};


    // Create CPU input tensor.
    Tensor host_input("host_input", shape);

    for(int i = 0; i < size; i++)
        host_input[static_cast<std::size_t>(i)] = static_cast<float>((i % 17 - 8) * 0.25f);

    

    // CPU Reference
    Tensor cpu_reference("cpu_reference", shape);

    gelu(host_input.data(), cpu_reference.data(), size);



    // Create an input on the GPU
    CudaMemoryPool input_cuda_pool;
    float* input_device_buffer = input_cuda_pool.allocate(host_input.size());

    Tensor device_input("input_gpu", shape, input_device_buffer, DeviceType::CUDA);



    // H2D Transfer
    copy_host_to_device(host_input, device_input);




    // Create graph
    Graph graph;

    Node gelu_node(
        "gelu_node", 
        "GELU", 
        {"input_gpu"},
        {"output_gpu"}
    );

    graph.add_node(gelu_node);

    graph.build_dependencies();




    // Configure Executor.
    Executor executor;

    executor.add_tensor(device_input);
    executor.register_tensor("output_gpu", shape, DeviceType::CUDA);




    // Run Graph
    executor.run(graph);


    executor.synchronize();


    // Obtain CUDA output tensor
    Tensor& device_output = executor.get_tensor("output_gpu");

    if(device_output.device() != DeviceType::CUDA){
        std::cerr << "Output Tensor is not on CUDA" << std::endl;

        input_cuda_pool.release(input_device_buffer);

        return 1;
    }


    // D2H
    Tensor host_output("host_output", shape);
    copy_device_to_host(device_output, host_output);    // 原来default stream的时候是在这里同步的，现在我们自己创立了stream，因此不能依赖隐形的保障关系





    // Compare CPU and GPU results
    bool correct = true;
    float max_diff = 0.0f;
    int max_diff_index = -1;

    for(int i = 0; i < size; i++){
        const float diff = std::fabs(cpu_reference[static_cast<std::size_t>(i)] - host_output[static_cast<std::size_t>(i)]);

        if(diff > max_diff){
            max_diff = diff;
            max_diff_index = i;
        }

        if(diff > tolerance)
            correct = false;
        
    }


    // Print result;
    std::cout << "Output Device: " << (device_output.device() == DeviceType::CUDA ? "CUDA" : "CPU") << std::endl;
    std::cout << "Max difference index: " << max_diff_index << std::endl;
    std::cout << "CUDA Executor GELU: " << (correct ? "PASS" : "FAIL") << std::endl;



    // Release external input CUDA memory.
    input_cuda_pool.release(input_device_buffer);

    return correct ? 0 : 1;

}