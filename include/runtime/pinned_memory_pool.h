// For CPU pinned memory

#ifndef PINNED_MEMORY_POOL_H
#define PINNED_MEMORY_POOL_H

#include <cstddef>
#include <vector>


struct PinnedMemoryBlock{
    float* data = nullptr;
    std::size_t capacity = 0;
    bool in_use = false;
};



class PinnedMemoryPool{
    public:

        float* allocate(std::size_t size);

        void release(float* ptr);


        std::size_t block_count() const;
        std::size_t total_capacity() const;

        ~PinnedMemoryPool();




    private:

        std::vector<PinnedMemoryBlock> blocks_;
};


#endif