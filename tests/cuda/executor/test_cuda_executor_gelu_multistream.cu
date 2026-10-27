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
#include <cstddef>
#include <iostream>
#include <vector>


// ==================================================
// Helper:
// Compare pinned CUDA output with CPU reference.
// ==================================================

bool compare_output(
    const PinnedTensor& actual,
    const Tensor& expected,
    float tolerance
)
{
    if(actual.shape() != expected.shape())
    {
        std::cerr
            << "Shape mismatch"
            << std::endl;

        return false;
    }


    float max_diff = 0.0f;

    std::size_t max_diff_index = 0;


    for(std::size_t i = 0; i < actual.size(); i++)
    {
        const float diff =
            std::fabs(
                actual[i]
                -
                expected[i]
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
            std::cerr
                << "Mismatch at index "
                << i
                << ": actual="
                << actual[i]
                << ", expected="
                << expected[i]
                << ", diff="
                << diff
                << std::endl;

            return false;
        }
    }


    std::cout
        << "Max difference: "
        << max_diff
        << " at index "
        << max_diff_index
        << std::endl;


    return true;
}


// ==================================================
// Main
// ==================================================

int main()
{
    const int size =
        1 << 20;


    const float tolerance =
        1e-5f;


    const std::vector<int> shape =
        {size};


    // ==================================================
    // Step 1:
    // Create pinned host input/output tensors.
    // ==================================================

    PinnedMemoryPool pinned_pool;


    PinnedTensor host_input_a(
        "host_input_a",
        shape,
        pinned_pool
    );


    PinnedTensor host_output_a(
        "host_output_a",
        shape,
        pinned_pool
    );


    PinnedTensor host_input_b(
        "host_input_b",
        shape,
        pinned_pool
    );


    PinnedTensor host_output_b(
        "host_output_b",
        shape,
        pinned_pool
    );


    // ==================================================
    // Step 2:
    // Fill two different input datasets.
    //
    // Important:
    // A and B intentionally use different values.
    // This helps detect accidental buffer sharing.
    // ==================================================

    for(int i = 0; i < size; i++)
    {
        host_input_a[
            static_cast<std::size_t>(i)
        ] =
            static_cast<float>(
                (i % 17 - 8)
                * 0.25f
            );


        host_input_b[
            static_cast<std::size_t>(i)
        ] =
            static_cast<float>(
                (i % 23 - 11)
                * 0.125f
            );
    }


    // ==================================================
    // Step 3:
    // CPU references.
    // ==================================================

    Tensor reference_a(
        "reference_a",
        shape
    );


    Tensor reference_b(
        "reference_b",
        shape
    );


    gelu(
        host_input_a.data(),
        reference_a.data(),
        size
    );


    gelu(
        host_input_b.data(),
        reference_b.data(),
        size
    );


    // ==================================================
    // Step 4:
    // Create two independent Executors.
    //
    // Each Executor owns its own CUDA stream.
    // ==================================================

    Executor executor_a;

    Executor executor_b;


    if(
        executor_a.get_stream()
        ==
        executor_b.get_stream()
    )
    {
        std::cerr
            << "ERROR: Both Executors unexpectedly "
            << "share the same CUDA stream."
            << std::endl;

        return 1;
    }


    std::cout
        << "Executor A stream: "
        << executor_a.get_stream()
        << std::endl;


    std::cout
        << "Executor B stream: "
        << executor_b.get_stream()
        << std::endl;


    // ==================================================
    // Step 5:
    // Allocate independent CUDA input buffers.
    // ==================================================

    CudaMemoryPool input_pool_a;

    CudaMemoryPool input_pool_b;


    float* device_input_ptr_a =
        input_pool_a.allocate(
            host_input_a.size()
        );


    float* device_input_ptr_b =
        input_pool_b.allocate(
            host_input_b.size()
        );


    Tensor device_input_a(
        "input_a",
        shape,
        device_input_ptr_a,
        DeviceType::CUDA
    );


    Tensor device_input_b(
        "input_b",
        shape,
        device_input_ptr_b,
        DeviceType::CUDA
    );


    // ==================================================
    // Step 6:
    // Build Graph A.
    // ==================================================

    Graph graph_a;


    graph_a.add_node(
        Node(
            "gelu_node_a",
            "GELU",
            {"input_a"},
            {"output_a"}
        )
    );


    graph_a.build_dependencies();


    // ==================================================
    // Step 7:
    // Build Graph B.
    // ==================================================

    Graph graph_b;


    graph_b.add_node(
        Node(
            "gelu_node_b",
            "GELU",
            {"input_b"},
            {"output_b"}
        )
    );


    graph_b.build_dependencies();


    // ==================================================
    // Step 8:
    // Configure Executor A.
    // ==================================================

    executor_a.add_tensor(
        device_input_a
    );


    executor_a.register_tensor(
        "output_a",
        shape,
        DeviceType::CUDA
    );


    // ==================================================
    // Step 9:
    // Configure Executor B.
    // ==================================================

    executor_b.add_tensor(
        device_input_b
    );


    executor_b.register_tensor(
        "output_b",
        shape,
        DeviceType::CUDA
    );


    // ==================================================
    // Step 10:
    // Submit complete pipeline A.
    //
    // Stream A:
    //
    // H2D A
    //   ↓
    // GELU A
    //   ↓
    // D2H A
    //
    // IMPORTANT:
    // Do NOT synchronize here.
    // ==================================================

    copy_host_to_device(
        host_input_a,
        device_input_a,
        executor_a.get_stream()
    );


    executor_a.run(
        graph_a
    );


    Tensor& device_output_a =
        executor_a.get_tensor(
            "output_a"
        );


    copy_device_to_host(
        device_output_a,
        host_output_a,
        executor_a.get_stream()
    );


    // ==================================================
    // Step 11:
    // Submit complete pipeline B immediately.
    //
    // Stream B:
    //
    // H2D B
    //   ↓
    // GELU B
    //   ↓
    // D2H B
    //
    // Still no synchronization.
    // ==================================================

    copy_host_to_device(
        host_input_b,
        device_input_b,
        executor_b.get_stream()
    );


    executor_b.run(
        graph_b
    );


    Tensor& device_output_b =
        executor_b.get_tensor(
            "output_b"
        );


    copy_device_to_host(
        device_output_b,
        host_output_b,
        executor_b.get_stream()
    );


    // ==================================================
    // Step 12:
    // Synchronize only after BOTH pipelines
    // have already been submitted.
    //
    // This allows CUDA to overlap independent
    // work from Stream A and Stream B.
    // ==================================================

    executor_a.synchronize();

    executor_b.synchronize();


    // ==================================================
    // Step 13:
    // Validate Pipeline A.
    // ==================================================

    std::cout
        << "\nChecking Pipeline A..."
        << std::endl;


    const bool correct_a =
        compare_output(
            host_output_a,
            reference_a,
            tolerance
        );


    // ==================================================
    // Step 14:
    // Validate Pipeline B.
    // ==================================================

    std::cout
        << "\nChecking Pipeline B..."
        << std::endl;


    const bool correct_b =
        compare_output(
            host_output_b,
            reference_b,
            tolerance
        );


    // ==================================================
    // Step 15:
    // Release external CUDA input blocks.
    // ==================================================

    input_pool_a.release(
        device_input_ptr_a
    );


    input_pool_b.release(
        device_input_ptr_b
    );


    // ==================================================
    // Step 16:
    // Final result.
    // ==================================================

    const bool correct =
        correct_a
        &&
        correct_b;


    std::cout
        << "\nMulti-stream GELU Pipeline A: "
        << (
            correct_a
            ? "PASS"
            : "FAIL"
        )
        << std::endl;


    std::cout
        << "Multi-stream GELU Pipeline B: "
        << (
            correct_b
            ? "PASS"
            : "FAIL"
        )
        << std::endl;


    std::cout
        << "\nCUDA Executor Multi-stream GELU Test: "
        << (
            correct
            ? "PASS"
            : "FAIL"
        )
        << std::endl;


    return correct ? 0 : 1;
}