#include "tensor/pinned_tensor.h"
#include <stdexcept>



// Helper
static std::size_t compute_size(const std::vector<int>& shape){
    std::size_t total_size = 1;

    for(int dim : shape){
        if(dim <= 0)
            throw std::invalid_argument("Dimension must be positive");

        total_size *= static_cast<size_t>(dim);
    }

    return total_size;
}


PinnedTensor::PinnedTensor(
    const std::string& name,
    const std::vector<int>& shape,
    PinnedMemoryPool& pool
) : name_(name), shape_(shape), size_(compute_size(shape)), pool_(&pool){
    
    data_ = pool_ -> allocate(size_);
}



PinnedTensor::~PinnedTensor(){
    if(pool_ != nullptr && data_ != nullptr){
        pool_ -> release(data_);

        data_ = nullptr;
    }
}



// 这边因为是 host memory， 所以可以直接[]调用。
float& PinnedTensor::operator[](std::size_t index){
    return data_[index];
}


const float& PinnedTensor::operator[](std::size_t index) const{
    return data_[index];
}






// Getter
const std::string& PinnedTensor::name() const {
    return name_;
}


const std::vector<int>& PinnedTensor::shape() const {
    return shape_;
}


std::size_t PinnedTensor::size() const {
    return size_;
}


float* PinnedTensor::data() {
    return data_;
}


const float* PinnedTensor::data() const {
    return data_;
}



