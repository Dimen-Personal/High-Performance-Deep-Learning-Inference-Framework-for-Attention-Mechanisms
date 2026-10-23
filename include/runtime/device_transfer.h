#ifndef DEVICE_TRANSFER_H
#define DEVICE_TRANSFER_H

#include "tensor/tensor.h"

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


#endif