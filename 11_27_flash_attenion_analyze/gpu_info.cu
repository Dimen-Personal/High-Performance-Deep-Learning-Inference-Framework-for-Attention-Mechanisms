
#include <cuda_runtime.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <string>


// ============================================================
// CUDA Error Checking
// ============================================================

void check_cuda(cudaError_t error)
{
    if(error != cudaSuccess)
    {
        throw std::runtime_error(
            cudaGetErrorString(error)
        );
    }
}


// ============================================================
// Print GPU Hardware Information
// ============================================================

void print_gpu_info(int device_id)
{
    cudaDeviceProp prop{};

    check_cuda(
        cudaGetDeviceProperties(
            &prop,
            device_id
        )
    );


    // ========================================================
    // 1. Basic Information
    // ========================================================

    std::cout
        << "\n========================================\n"
        << "GPU Hardware Information\n"
        << "========================================\n";

    std::cout
        << "Device ID: "
        << device_id << '\n';

    std::cout
        << "GPU Name: "
        << prop.name << '\n';

    std::cout
        << "Compute Capability: "
        << prop.major << "."
        << prop.minor << '\n';


    // ========================================================
    // 2. Compute Resources
    // ========================================================

    std::cout
        << "\n===== Compute Resources =====\n";

    std::cout
        << "SM Count: "
        << prop.multiProcessorCount << '\n';

    std::cout
        << "Warp Size: "
        << prop.warpSize << '\n';

    std::cout
        << "Max Threads per Block: "
        << prop.maxThreadsPerBlock << '\n';

    std::cout
        << "Max Threads per SM: "
        << prop.maxThreadsPerMultiProcessor << '\n';

    std::cout
        << "Max Warps per SM: "
        << prop.maxThreadsPerMultiProcessor
           / prop.warpSize
        << '\n';


    // ========================================================
    // 3. Global Memory and L2 Cache
    // ========================================================

    std::cout
        << "\n===== Global Memory =====\n";

    std::cout
        << "Global Memory: "
        << prop.totalGlobalMem
           / (1024.0 * 1024.0 * 1024.0)
        << " GiB\n";

    std::cout
        << "L2 Cache Size: "
        << prop.l2CacheSize
           / (1024.0 * 1024.0)
        << " MiB\n";

    std::cout
        << "Memory Bus Width: "
        << prop.memoryBusWidth
        << " bits\n";


    // ========================================================
    // 4. Shared Memory
    // ========================================================

    std::cout
        << "\n===== Shared Memory =====\n";

    std::cout
        << "Shared Memory per Block: "
        << prop.sharedMemPerBlock
           / 1024.0
        << " KiB\n";

    std::cout
        << "Opt-in Shared Memory per Block: "
        << prop.sharedMemPerBlockOptin
           / 1024.0
        << " KiB\n";

    std::cout
        << "Shared Memory per SM: "
        << prop.sharedMemPerMultiprocessor
           / 1024.0
        << " KiB\n";


    // ========================================================
    // 5. Register Resources
    // ========================================================

    std::cout
        << "\n===== Register Resources =====\n";

    std::cout
        << "Registers per Block: "
        << prop.regsPerBlock << '\n';

    std::cout
        << "Registers per SM: "
        << prop.regsPerMultiprocessor << '\n';


    // ========================================================
    // 6. Maximum Resident Blocks
    // ========================================================

    int max_blocks_per_sm = 0;

    check_cuda(
        cudaDeviceGetAttribute(
            &max_blocks_per_sm,
            cudaDevAttrMaxBlocksPerMultiprocessor,
            device_id
        )
    );

    std::cout
        << "\n===== Block Resources =====\n";

    std::cout
        << "Max Blocks per SM: "
        << max_blocks_per_sm << '\n';
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        int device_count = 0;

        check_cuda(
            cudaGetDeviceCount(
                &device_count
            )
        );

        std::cout
            << std::fixed
            << std::setprecision(2);

        std::cout
            << "Available CUDA Devices: "
            << device_count << '\n';

        if(device_count == 0)
        {
            std::cerr
                << "No CUDA devices found.\n";

            return 1;
        }

        for(int i = 0; i < device_count; i++)
        {
            print_gpu_info(i);
        }

        return 0;
    }
    catch(const std::exception& exception)
    {
        std::cerr
            << "CUDA Error: "
            << exception.what()
            << '\n';

        return 1;
    }
}