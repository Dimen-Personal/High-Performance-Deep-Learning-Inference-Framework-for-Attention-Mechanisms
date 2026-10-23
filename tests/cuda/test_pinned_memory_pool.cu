#include "runtime/pinned_memory_pool.h"
#include "runtime/cuda_memory_pool.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "runtime/pinned_memory_pool.h"

#include <iostream>


void pinned_pool_basic()
{
    PinnedMemoryPool pool;


    const std::size_t size = 1024;


    float* ptr =
        pool.allocate(
            size
        );


    for(std::size_t i = 0; i < size; i++)
    {
        ptr[i] =
            static_cast<float>(i);
    }


    for(std::size_t i = 0; i < size; i++)
    {
        if(
            ptr[i] !=
            static_cast<float>(i)
        )
        {
            std::cerr
                << "Pinned memory access failed"
                << std::endl;

            return ;
        }
    }


    pool.release(
        ptr
    );


    std::cout
        << "Pinned Memory Pool: PASS"
        << std::endl;


     
}



void pinned_pool_transfer()
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

            return ;
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


    
}



void pinned_pool_reuse()
{
    PinnedMemoryPool pool;


    // First allocation.
    float* first =
        pool.allocate(
            1024
        );


    const std::size_t block_count_before =
        pool.block_count();


    // Release the block back to the pool.
    pool.release(
        first
    );


    // Request a smaller block.
    // Since the previous block has capacity 1024,
    // it should be reused for this 512-element request.
    float* second =
        pool.allocate(
            512
        );


    const std::size_t block_count_after =
        pool.block_count();


    // The same memory block should be reused.
    if(first != second)
    {
        std::cerr
            << "Pinned memory reuse failed: "
            << "different pointer returned"
            << std::endl;

        return;
    }


    // No new block should be created.
    if(block_count_before != block_count_after)
    {
        std::cerr
            << "Pinned memory reuse failed: "
            << "new block was allocated"
            << std::endl;

        return;
    }


    pool.release(
        second
    );


    std::cout
        << "Pinned Memory Reuse: PASS"
        << std::endl;
}



int main(){
    pinned_pool_basic();
    pinned_pool_reuse();
    pinned_pool_transfer();


    return 0;
}