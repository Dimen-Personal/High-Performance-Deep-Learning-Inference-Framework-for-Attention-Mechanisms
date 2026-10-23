#include "runtime/pinned_memory_pool.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>


float* PinnedMemoryPool::allocate(std::size_t size){
   
   if(size == 0)
        throw std::invalid_argument("PinnedMemoryPool cannot allocate 0 elements");
    


    // Try to reuse a free block.
    for(PinnedMemoryBlock& block : blocks_)
        if(!block.in_use && block.capacity >= size){
            block.in_use = true;

            return block.data;
        }
    


    // No reusable block.
    PinnedMemoryBlock block;


    cudaError_t status = cudaMallocHost(
        reinterpret_cast<void**>(&block.data),
        size * sizeof(float)
    );



    if(status != cudaSuccess){
        
        throw std::runtime_error(std::string("cudaMallocHost failed: ")
            +
            cudaGetErrorString(status)
        );
    }


    block.capacity = size;
    block.in_use = true;


    blocks_.push_back(
        block
    );


    return block.data;
}




// Release Operation * this is only release (not destroy) one block"
void PinnedMemoryPool::release(float* ptr){
    if(ptr == nullptr)
        throw std::invalid_argument("cannot release null pointer");

    for(PinnedMemoryBlock& block : blocks_){
        if(block.data == ptr){
            if(!block.in_use)
                throw std::runtime_error("Pinned Memory Block is already free");

            block.in_use = false;

            return;
        }
    }


    throw std::runtime_error("Pointer does not belong to PinnedMemoryPool");
}




// Deconstructor
PinnedMemoryPool::~PinnedMemoryPool(){
    for(PinnedMemoryBlock& block : blocks_){
        if(block.data != nullptr){
            cudaFreeHost(block.data);

            block.data = nullptr;
        }
    }
}




std::size_t PinnedMemoryPool::block_count() const{
    return blocks_.size();
}




std::size_t PinnedMemoryPool::total_capacity() const{
    
    std::size_t total = 0;

    for(const PinnedMemoryBlock& block : blocks_)
        total += block.capacity;
    
    return total;
}