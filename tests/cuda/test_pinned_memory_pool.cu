#include "runtime/pinned_memory_pool.h"
#include "runtime/cuda_memory_pool.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>


int main()
{
    const std::size_t size = 1024;


    PinnedMemoryPool pinned_pool;
    CudaMemoryPool cuda_pool;


    float* host_input =
        pinned_pool.allocate(
            size
        );


    float* host_output =
        pinned_pool.allocate(
            size
        );


    float* device =
        cuda_pool.allocate(
            size
        );


    for(std::size_t i = 0; i < size; i++)
    {
        host_input[i] =
            static_cast<float>(i);
    }


    cudaStream_t stream;

    cudaStreamCreate(
        &stream
    );


    cudaMemcpyAsync(
        device,
        host_input,
        size * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    );


    cudaMemcpyAsync(
        host_output,
        device,
        size * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    );


    cudaStreamSynchronize(
        stream
    );


    for(std::size_t i = 0; i < size; i++)
    {
        if(
            host_input[i]
            !=
            host_output[i]
        )
        {
            std::cerr
                << "Async copy mismatch"
                << std::endl;

            return 1;
        }
    }


    cudaStreamDestroy(
        stream
    );


    pinned_pool.release(
        host_input
    );

    pinned_pool.release(
        host_output
    );

    cuda_pool.release(
        device
    );


    std::cout
        << "Pinned Async Transfer: PASS"
        << std::endl;


    return 0;
}