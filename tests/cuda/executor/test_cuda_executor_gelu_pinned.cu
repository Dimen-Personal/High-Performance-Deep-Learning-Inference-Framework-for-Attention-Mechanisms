#include "graph/graph.h"
#include "graph/node.h"

#include "ops/operator.h"

#include "tensor/tensor.h"
#include "tensor/pinned_tensor.h"

#include "runtime/cuda_memory_pool.h"
#include "runtime/device_transfer.h"
#include "runtime/executor.h"
#include "runtime/pinned_memory_pool.h"

#include <cmath>
#include <iostream>
#include <vector>


int main()
{
    const int size = 513;

    const float tolerance =
        1e-5f;

    const std::vector<int> shape =
        {size};


    // ==================================================
    // 1. Create pinned host input/output buffers.
    // ==================================================

    PinnedMemoryPool pinned_pool;


    PinnedTensor host_input(
        "host_input",
        shape,
        pinned_pool
    );


    PinnedTensor host_output(
        "host_output",
        shape,
        pinned_pool
    );


    for(int i = 0; i < size; i++)
    {
        host_input[
            static_cast<std::size_t>(i)
        ] =
            static_cast<float>(
                (i % 17 - 8) * 0.25f
            );
    }


    // ==================================================
    // 2. Create ordinary CPU reference.
    // ==================================================

    Tensor cpu_reference(
        "cpu_reference",
        shape
    );


    gelu(
        host_input.data(),
        cpu_reference.data(),
        size
    );


    // ==================================================
    // 3. Allocate CUDA input buffer.
    // ==================================================

    CudaMemoryPool cuda_input_pool;


    float* device_input_buffer =
        cuda_input_pool.allocate(
            host_input.size()
        );


    Tensor device_input(
        "input_gpu",
        shape,
        device_input_buffer,
        DeviceType::CUDA
    );


    // ==================================================
    // 4. Create Executor.
    //
    // Executor owns the CUDA stream used by:
    //
    // H2D
    // GELU
    // D2H
    // ==================================================

    Executor executor;


    // ==================================================
    // 5. Async H2D from pinned host memory.
    // ==================================================

    copy_host_to_device(
        host_input,
        device_input,
        executor.get_stream()
    );


    // ==================================================
    // 6. Build GELU graph.
    // ==================================================

    Graph graph;


    graph.add_node(
        Node(
            "gelu_node",
            "GELU",
            {"input_gpu"},
            {"output_gpu"}
        )
    );


    graph.build_dependencies();


    // ==================================================
    // 7. Configure Executor tensors.
    // ==================================================

    executor.add_tensor(
        device_input
    );


    executor.register_tensor(
        "output_gpu",
        shape,
        DeviceType::CUDA
    );


    // ==================================================
    // 8. Enqueue CUDA GELU.
    // ==================================================

    executor.run(
        graph
    );


    // ==================================================
    // 9. Obtain device output metadata.
    //
    // This does NOT synchronize the GPU.
    // ==================================================

    Tensor& device_output =
        executor.get_tensor(
            "output_gpu"
        );


    if(
        device_output.device()
        != DeviceType::CUDA
    )
    {
        std::cerr
            << "Output tensor is not on CUDA"
            << std::endl;

        cuda_input_pool.release(
            device_input_buffer
        );

        return 1;
    }


    // ==================================================
    // 10. Async D2H into pinned host memory.
    //
    // Same stream guarantees:
    //
    // H2D
    //  ↓
    // GELU
    //  ↓
    // D2H
    // ==================================================

    copy_device_to_host(
        device_output,
        host_output,
        executor.get_stream()
    );


    // ==================================================
    // 11. Synchronize once.
    //
    // Wait for:
    //
    // H2D + GELU + D2H
    // ==================================================

    executor.synchronize();


    // ==================================================
    // 12. Compare CPU and GPU results.
    // ==================================================

    bool correct = true;

    float max_diff =
        0.0f;

    int max_diff_index =
        -1;


    for(int i = 0; i < size; i++)
    {
        const float cpu_value =
            cpu_reference[
                static_cast<std::size_t>(i)
            ];


        const float gpu_value =
            host_output[
                static_cast<std::size_t>(i)
            ];


        const float diff =
            std::fabs(
                cpu_value
                -
                gpu_value
            );


        if(diff > max_diff)
        {
            max_diff =
                diff;

            max_diff_index =
                i;
        }


        if(diff > tolerance)
        {
            correct = false;
        }
    }


    // ==================================================
    // 13. Print result.
    // ==================================================

    std::cout
        << "Output Device: "
        << (
            device_output.device()
            == DeviceType::CUDA
            ? "CUDA"
            : "CPU"
        )
        << std::endl;


    std::cout
        << "Max difference: "
        << max_diff
        << std::endl;


    std::cout
        << "Max difference index: "
        << max_diff_index
        << std::endl;


    std::cout
        << "Pinned Async CUDA Executor GELU: "
        << (
            correct
            ? "PASS"
            : "FAIL"
        )
        << std::endl;


    // ==================================================
    // 14. Return external CUDA input block to pool.
    // ==================================================

    cuda_input_pool.release(
        device_input_buffer
    );


    return correct ? 0 : 1;
}