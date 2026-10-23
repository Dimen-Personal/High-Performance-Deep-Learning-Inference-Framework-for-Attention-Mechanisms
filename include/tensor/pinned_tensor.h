#ifndef PINNED_TENSOR_H
#define PINNED_TENSOR_H

#include "runtime/pinned_memory_pool.h"

#include <cstddef>
#include <string>
#include <vector>


class PinnedTensor{

    public:

        PinnedTensor(
            const std::string& name,
            const std::vector<int>& shape,
            PinnedMemoryPool& pool
        );


        ~PinnedTensor();


        PinnedTensor(const PinnedTensor&) = delete;
        PinnedTensor& operator=(const PinnedTensor&) = delete;


        const std::string& name() const;
        const std::vector<int>& shape() const;
        std::size_t size() const;


        float* data();
        const float* data() const;


        float& operator[](std::size_t index);
        const float& operator[](std::size_t index) const;


    
    
    private:

        std::string name_;
        std::vector<int> shape_;
        std::size_t size_ = 0; 


}