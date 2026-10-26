#ifndef DEVICE_TRANSFER_H
#define DEVICE_TRANSFER_H

#include "tensor/tensor.h"
#include "tensor/pinned_tensor.h"
#include <cuda_runtime.h>

void copy_host_to_device(
    const Tensor& host,
    Tensor& device,
    cudaStream_t stream = nullptr
);


void copy_device_to_host(
    const Tensor& device,
    Tensor& host,
    cudaStream_t stream = nullptr
);



// Overload
void copy_host_to_device(
    const PinnedTensor& host,
    Tensor& device,
    cudaStream_t stream = nullptr
);

void copy_device_to_host(
    const Tensor& device,
    PinnedTensor& host,
    cudaStream_t stream = nullptr
);

#endif