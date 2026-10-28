cmake_minimum_required(VERSION 3.20)

project(
    HighPerformanceInference       # name
    VERSION 1.0
    LANGUAGES CXX CUDA
)

# since cuda_ops.h is cpp file, so now we need to let cpp know where can find cudatoolkit.
find_package(CUDAToolkit REQUIRED)


set(CMAKE_CXX_STANDARD 17)          # using C++ 17 standard  == g++ -std=c++17
set(CMAKE_CXX_STANDARD_REQUIRED ON) # Necessary C++ 17

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# OpenMP on macOS
if(APPLE)
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)



# nlohmann JSON
find_package(nlohmann_json REQUIRED)





# Make a variable OPT_LEVEL for CMake
set(
    OPT_LEVEL
    "O0"
    CACHE STRING  # Set default
    "Compiler optimization level: O0, O1, O2, O3, Ofast" # Explanation for colleges
)

# Update complier based on the optimization level
if(OPT_LEVEL STREQUAL "O0")
    add_compile_options(-O0)
elseif(OPT_LEVEL STREQUAL "O1")
    add_compile_options(-O1)
elseif(OPT_LEVEL STREQUAL "O2")
    add_compile_options(-O2)
elseif(OPT_LEVEL STREQUAL "O3")
    add_compile_options(-O3)
elseif(OPT_LEVEL STREQUAL "Ofast")
    add_compile_options(-O3 -ffast-math)

else()
    message(
        FATAL_ERROR
        "Invalid OPT_LEVEL: ${OPT_LEVEL}"
    )
endif()


message(
    STATUS
    "Complier optimization level: ${OPT_LEVEL}"
)





include_directories(
    include
    /opt/homebrew/opt/libomp/include
)


# For correctness
add_executable(
    gemm_test

    tests/cpp/test_gemm.cpp
    src/utils/matrix.cpp

    src/gemm/naive_gemm.cpp
    src/gemm/reordered_gemm.cpp
    src/gemm/unrolled_gemm.cpp
    src/gemm/blocked_gemm.cpp
    src/gemm/blocked2_gemm.cpp
    src/gemm/simd_gemm.cpp
    src/gemm/simd_blocked_gemm.cpp
    src/gemm/optimized_gemm.cpp
)

target_include_directories(
    gemm_test
    PRIVATE
    /opt/homebrew/opt/libomp/include
)


target_link_libraries(
    gemm_test
    PRIVATE
    OpenMP::OpenMP_CXX
)


# For efficiency
add_executable(
    gemm_benchmark

    benchmark/benchmark_gemm.cpp
    src/gemm/naive_gemm.cpp
    src/gemm/reordered_gemm.cpp
    src/gemm/unrolled_gemm.cpp
    src/gemm/blocked_gemm.cpp
    src/gemm/blocked2_gemm.cpp
    src/gemm/simd_gemm.cpp
    src/gemm/simd_blocked_gemm.cpp
    src/gemm/optimized_gemm.cpp
    
    src/utils/matrix.cpp
    src/utils/timer.cpp
)

target_include_directories(
    gemm_benchmark
    PRIVATE
    /opt/homebrew/opt/libomp/include
)


target_link_libraries(
    gemm_benchmark
    PRIVATE
    OpenMP::OpenMP_CXX
)





# For Operator Tests
add_executable(
    ops_test

    tests/cpp/test_ops.cpp
    #tests/cpp/test_ops_temp.cpp

    src/ops/attention/softmax.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/gelu.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp
)


# For Graph basic test
add_executable(
    graph_test

    tests/cpp/test_graph.cpp

    src/tensor/tensor.cpp
    src/graph/graph.cpp
    src/graph/node.cpp
)


target_link_libraries(
    graph_test
    PRIVATE
    nlohmann_json::nlohmann_json
)



# For Graph Parser test
add_executable(
    parser_test

    tests/cpp/test_parser.cpp

    src/tensor/tensor.cpp
    src/graph/graph.cpp
    src/graph/node.cpp
    src/graph/parser.cpp
)

target_link_libraries(
    parser_test
    PRIVATE
    nlohmann_json::nlohmann_json
)


# For Graph Executor Test
add_executable(
    executor_test

    tests/cpp/test_executor.cpp
    
    src/tensor/tensor.cpp

    src/runtime/executor.cpp
    src/runtime/memory_planner.cpp
    src/runtime/memory_pool.cpp

    src/graph/graph.cpp
    src/graph/node.cpp
    src/graph/parser.cpp
  

    src/ops/attention/softmax.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/gelu.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp
)

target_link_libraries(
    executor_test
    PRIVATE
    nlohmann_json::nlohmann_json
)




# For Memory Planner Test
add_executable(
    memory_planner_test

    tests/cpp/test_memory_planner.cpp
    
    src/tensor/tensor.cpp

    src/graph/graph.cpp
    src/graph/node.cpp
    src/graph/parser.cpp
    
    src/runtime/executor.cpp
    src/runtime/memory_planner.cpp
    src/runtime/memory_pool.cpp

    src/ops/attention/softmax.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/gelu.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp
)

target_link_libraries(
    memory_planner_test
    PRIVATE
    nlohmann_json::nlohmann_json
)






# For Memory Pool Test
add_executable(
    memory_pool_test

    tests/cpp/test_memory_pool.cpp
    
    src/tensor/tensor.cpp

    src/graph/graph.cpp
    src/graph/node.cpp
    src/graph/parser.cpp
    
    src/runtime/executor.cpp
    src/runtime/memory_planner.cpp
    src/runtime/memory_pool.cpp

    src/ops/attention/softmax.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/gelu.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp
)

target_link_libraries(
    memory_pool_test
    PRIVATE
    nlohmann_json::nlohmann_json
)




# For full Memory Pool execute Test
add_executable(
    execute_memory_pool_test

    tests/cpp/test_memory_aware_executor.cpp
    
    src/tensor/tensor.cpp

    src/graph/graph.cpp
    src/graph/node.cpp
    src/graph/parser.cpp
    
    src/runtime/executor.cpp
    src/runtime/memory_planner.cpp
    src/runtime/memory_pool.cpp

    src/ops/attention/softmax.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/gelu.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp
)

target_link_libraries(
    execute_memory_pool_test
    PRIVATE
    nlohmann_json::nlohmann_json
)



# Next is the part for the CUDA
add_executable(
    cuda_vector_add_test

    tests/cuda/test_vector_add.cu
    src/cuda/vector_add.cu
)

target_include_directories(
    cuda_vector_add_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_property(
    TARGET cuda_vector_add_test
    PROPERTY CUDA_ARCHITECTURES 70   # Since V100S is Volta and compute capability 7.0, therefore let nvcc use sm_70 to compile.
)



# For cuda_gemm_test.
add_executable(
    cuda_gemm_test

    tests/cuda/test_cuda_gemm.cu
    src/cuda/gemm/gemm_naive.cu
    src/cuda/gemm/gemm_reordered.cu
    src/cuda/gemm/gemm_unrolled.cu
)

target_include_directories(
    cuda_gemm_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_property(
    TARGET cuda_gemm_test
    PROPERTY CUDA_ARCHITECTURES 70   
)



# For cuda_gemm_benchmark
add_executable(
    cuda_gemm_benchmark

    benchmark/cuda/benchmark_cuda_gemm.cu
    src/cuda/gemm/gemm_naive.cu
    src/cuda/gemm/gemm_reordered.cu
    src/cuda/gemm/gemm_unrolled.cu
)

target_include_directories(
    cuda_gemm_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_property(
    TARGET cuda_gemm_benchmark
    PROPERTY CUDA_ARCHITECTURES 70
)



# For cuda_ops_test
add_executable(
    cuda_ops_test

    tests/cuda/test_cuda_ops.cu

    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp

    src/cuda/ops/gelu.cu
    src/cuda/ops/layernorm.cu
    src/cuda/ops/softmax.cu 
)

target_include_directories(
    cuda_ops_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_property(
    TARGET cuda_ops_test
    PROPERTY CUDA_ARCHITECTURES 70
)




# For cuda_memory_pool_test
add_executable(
    cuda_memory_pool_test

    tests/cuda/test_cuda_memory_pool.cu

    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp

    src/runtime/cuda_memory_pool.cu

    src/cuda/ops/gelu.cu
    src/cuda/ops/layernorm.cu
    src/cuda/ops/softmax.cu 

)

target_include_directories(
    cuda_memory_pool_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_property(
    TARGET cuda_memory_pool_test
    PROPERTY CUDA_ARCHITECTURES 70
)



# For device_transfer_test
add_executable(
    device_transfer_test

    tests/cuda/test_device_transfer.cu

    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp

    src/tensor/tensor.cpp

    src/runtime/cuda_memory_pool.cu
    src/runtime/device_transfer.cu

    src/cuda/ops/gelu.cu
    src/cuda/ops/layernorm.cu
    src/cuda/ops/softmax.cu 

)

target_include_directories(
    device_transfer_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

set_property(
    TARGET device_transfer_test
    PROPERTY CUDA_ARCHITECTURES 70
)



# cuda_executor_gelu_test
add_executable(
    cuda_executor_gelu_test

    tests/cuda/executor/test_cuda_executor_gelu.cu

    src/tensor/tensor.cpp

    src/graph/node.cpp
    src/graph/graph.cpp

    src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp
    src/runtime/cuda_memory_pool.cu
    src/runtime/device_transfer.cu

    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp

    src/cuda/ops/gelu.cu
    src/cuda/ops/layernorm.cu
    src/cuda/ops/softmax.cu
)


target_include_directories(
    cuda_executor_gelu_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


set_property(
    TARGET cuda_executor_gelu_test
    PROPERTY CUDA_ARCHITECTURES 70
)





# cuda_matmul_test
add_executable(
    cuda_matmul_test

    tests/cuda/test_cuda_matmul.cu

    src/tensor/tensor.cpp

    src/graph/node.cpp
    src/graph/graph.cpp

    src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp
    src/runtime/cuda_memory_pool.cu
    src/runtime/device_transfer.cu

    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp

    src/cuda/ops/gelu.cu
    src/cuda/ops/layernorm.cu
    src/cuda/ops/softmax.cu
    src/cuda/ops/matmul.cu
)


target_include_directories(
    cuda_matmul_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


set_property(
    TARGET cuda_matmul_test
    PROPERTY CUDA_ARCHITECTURES 70
)




# cuda_attention_scores_test
add_executable(
    cuda_attention_scores_test

    tests/cuda/test_cuda_attention_scores.cu

    src/tensor/tensor.cpp

    src/graph/node.cpp
    src/graph/graph.cpp

    src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp
    src/runtime/cuda_memory_pool.cu
    src/runtime/device_transfer.cu

    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp

    src/cuda/ops/gelu.cu
    src/cuda/ops/layernorm.cu
    src/cuda/ops/softmax.cu
    src/cuda/ops/matmul.cu
    src/cuda/ops/attention_scores.cu
)


target_include_directories(
    cuda_attention_scores_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


set_property(
    TARGET cuda_attention_scores_test
    PROPERTY CUDA_ARCHITECTURES 70
)




# cuda_attention_softmax_test
add_executable(
    cuda_attention_softmax_test

    tests/cuda/test_cuda_attention_softmax.cu

    src/tensor/tensor.cpp

    src/graph/node.cpp
    src/graph/graph.cpp

    # src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp
    src/runtime/cuda_memory_pool.cu
    src/runtime/device_transfer.cu

    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp

 
    src/cuda/ops/gelu.cu
    src/cuda/ops/layernorm.cu
    src/cuda/ops/softmax.cu
    src/cuda/ops/matmul.cu
    src/cuda/ops/attention_scores.cu
    src/cuda/ops/attention_softmax.cu
    src/cuda/ops/attention_softmax_multiblock.cu
)


target_include_directories(
    cuda_attention_softmax_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


set_property(
    TARGET cuda_attention_softmax_test
    PROPERTY CUDA_ARCHITECTURES 70
)


# We can create some libraries on the CMake
add_library(
    cuda_backend STATIC   # 和 Shared对应，一般用于插件，共享组件等的研发，多个程序可以使用同一个库。

    src/runtime/cuda_memory_pool.cu
    src/runtime/device_transfer.cu

    src/cuda/ops/gelu.cu
    src/cuda/ops/layernorm.cu
    src/cuda/ops/softmax.cu
    src/cuda/ops/matmul.cu
    src/cuda/ops/attention_scores.cu
    src/cuda/ops/attention_softmax.cu
    src/cuda/ops/attention_softmax_multiblock.cu
)

target_include_directories(
    cuda_backend
    PUBLIC         # PUBLIC  PRIVATE   INTERFACE  规定了谁会使用这些头文件
    ${CMAKE_SOURCE_DIR}/include 
)


target_link_libraries(
    cuda_backend
    PUBLIC
    CUDA::cudart
)

set_property(
    TARGET cuda_backend
    PROPERTY CUDA_ARCHITECTURES 70
)



# CUDA Executor Attention Tests
add_executable(
    cuda_executor_attention_test

    # Test
    tests/cuda/executor/test_cuda_executor_attention.cu

    # Tensor
    src/tensor/tensor.cpp

    # Graph
    src/graph/node.cpp
    src/graph/graph.cpp

    # Runtime
    src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp

    # Original CPU Kernels
    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp

    # Operator Registry
    src/ops/operator_registry.cpp
    #src/ops/register_operators.cpp

    # CPU Polymorphic Operators
    src/ops/cpu/gelu_operator.cpp
    src/ops/cpu/matmul_operator.cpp
    src/ops/cpu/layernorm_operator.cpp
    src/ops/cpu/attention_scores_operator.cpp
    src/ops/cpu/attention_softmax_operator.cpp

    # CUDA Polymorphic Operators
    src/ops/cuda/gelu_operator.cpp
    src/ops/cuda/matmul_operator.cpp
    src/ops/cuda/layernorm_operator.cpp
    src/ops/cuda/attention_scores_operator.cpp
    src/ops/cuda/attention_softmax_operator.cpp
)

target_include_directories(
    cuda_executor_attention_test
    PRIVATE    # 这里如果当前文件只是一个中转站，那么PUBLIC就可以让后面的文件链接到这个依赖，现在是没有区别的，因为现在就是终点。
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(
    cuda_executor_attention_test
    PRIVATE
    cuda_backend
)


set_property(
    TARGET cuda_executor_attention_test
    PROPERTY CUDA_ARCHITECTURES 70
)





# Operator Registry Test
add_executable(
    operator_registry_test

    tests/cpp/test_operator_registry.cpp

    src/tensor/tensor.cpp

    src/ops/operator_registry.cpp
    src/ops/cpu/gelu_operator.cpp
    src/ops/attention/gelu.cpp
)

target_include_directories(
    operator_registry_test
    PRIVATE     
    ${CMAKE_SOURCE_DIR}/include
)
 
set_property(
    TARGET cuda_executor_attention_test
    PROPERTY CUDA_ARCHITECTURES 70
)



# CUDA Attention Softmax Benchmark
add_executable(
    cuda_attention_softmax_benchmark

    benchmark/cuda/benchmark_cuda_attention_softmax.cu

    src/cuda/ops/attention_softmax.cu
    src/cuda/ops/attention_softmax_multiblock.cu
)


target_include_directories(
    cuda_attention_softmax_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


set_property(
    TARGET cuda_attention_softmax_benchmark
    PROPERTY CUDA_ARCHITECTURES 70
)





# CUDA Attention Softmax Profile
add_executable(
    cuda_attention_softmax_profile

    benchmark/cuda/benchmark_profile_cuda_attention_softmax.cu

    src/cuda/ops/attention_softmax.cu
    src/cuda/ops/attention_softmax_multiblock.cu
)


target_include_directories(
    cuda_attention_softmax_profile
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


set_property(
    TARGET cuda_attention_softmax_profile
    PROPERTY CUDA_ARCHITECTURES 70
)




# Pinned memory Pool test
add_executable(
    pinned_memory_pool_test

    tests/cuda/test_pinned_memory_pool.cu

    src/runtime/pinned_memory_pool.cu
    src/runtime/cuda_memory_pool.cu
)


target_include_directories(
    pinned_memory_pool_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(
    pinned_memory_pool_test
    PRIVATE
    CUDA::cudart
)


set_property(
    TARGET pinned_memory_pool_test
    PROPERTY CUDA_ARCHITECTURES 70
)




# cuda_executor_gelu_pinned_test
add_executable(
    cuda_executor_gelu_pinned_test

    tests/cuda/executor/test_cuda_executor_gelu_pinned.cu

    # Tensor
    src/tensor/tensor.cpp
    src/tensor/pinned_tensor.cpp

    # Graph
    src/graph/node.cpp
    src/graph/graph.cpp

    # Runtime
    src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp
    src/runtime/cuda_memory_pool.cu
    src/runtime/pinned_memory_pool.cu
    src/runtime/device_transfer.cu

    # CPU GELU reference
    src/ops/attention/gelu.cpp

    # Operator registry
    src/ops/operator_registry.cpp

    # Polymorphic operators
    src/ops/cpu/gelu_operator.cpp
    src/ops/cuda/gelu_operator.cpp

    # CUDA GELU kernel
    src/cuda/ops/gelu.cu
)


target_include_directories(
    cuda_executor_gelu_pinned_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(
    cuda_executor_gelu_pinned_test
    PRIVATE
    CUDA::cudart
)


set_property(
    TARGET cuda_executor_gelu_pinned_test
    PROPERTY CUDA_ARCHITECTURES 70
)






# CUDA Executor Multi-stream GELU Test
add_executable(
    cuda_executor_gelu_multistream_test

    tests/cuda/executor/test_cuda_executor_gelu_multistream.cu

    # Tensor
    src/tensor/tensor.cpp
    src/tensor/pinned_tensor.cpp

    # Graph
    src/graph/node.cpp
    src/graph/graph.cpp

    # Runtime
    src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp
    src/runtime/cuda_memory_pool.cu
    src/runtime/pinned_memory_pool.cu
    src/runtime/device_transfer.cu

    # CPU GELU reference
    src/ops/attention/gelu.cpp

    # Operator registry
    src/ops/operator_registry.cpp

    # Operators
    src/ops/cpu/gelu_operator.cpp
    src/ops/cuda/gelu_operator.cpp

    # CUDA kernel
    src/cuda/ops/gelu.cu
)


target_include_directories(
    cuda_executor_gelu_multistream_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(
    cuda_executor_gelu_multistream_test
    PRIVATE
    CUDA::cudart
)


set_property(
    TARGET cuda_executor_gelu_multistream_test
    PROPERTY CUDA_ARCHITECTURES 70
)






# End-to-End Attention Multi-stream Benchmark
add_executable(
    cuda_executor_attention_multistream_benchmark

    benchmark/cuda/benchmark_executor_attention_multistream.cu

    # Tensor
    src/tensor/tensor.cpp
    src/tensor/pinned_tensor.cpp

    # Graph
    src/graph/node.cpp
    src/graph/graph.cpp

    # Runtime
    src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp
    src/runtime/cuda_memory_pool.cu
    src/runtime/pinned_memory_pool.cu
    src/runtime/device_transfer.cu

    # CPU kernels
    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp

    # Operator Registry
    src/ops/operator_registry.cpp

    # CPU operators
    src/ops/cpu/gelu_operator.cpp
    src/ops/cpu/matmul_operator.cpp
    src/ops/cpu/layernorm_operator.cpp
    src/ops/cpu/attention_scores_operator.cpp
    src/ops/cpu/attention_softmax_operator.cpp

    # CUDA operators
    src/ops/cuda/gelu_operator.cpp
    src/ops/cuda/matmul_operator.cpp
    src/ops/cuda/layernorm_operator.cpp
    src/ops/cuda/attention_scores_operator.cpp
    src/ops/cuda/attention_softmax_operator.cpp
)


target_include_directories(
    cuda_executor_attention_multistream_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(
    cuda_executor_attention_multistream_benchmark
    PRIVATE
    cuda_backend
    CUDA::cudart
)


set_property(
    TARGET cuda_executor_attention_multistream_benchmark
    PROPERTY CUDA_ARCHITECTURES 70
)








# CUDA Executor Multi-stream Attention Profile
add_executable(
    cuda_executor_attention_multistream_profile

    benchmark/cuda/profile_executor_attention_multistream.cu

    # Tensor
    src/tensor/tensor.cpp
    src/tensor/pinned_tensor.cpp

    # Graph
    src/graph/node.cpp
    src/graph/graph.cpp

    # Runtime
    src/runtime/executor.cpp
    src/runtime/memory_pool.cpp
    src/runtime/memory_planner.cpp
    src/runtime/cuda_memory_pool.cu
    src/runtime/pinned_memory_pool.cu
    src/runtime/device_transfer.cu

    # CPU kernels
    src/ops/attention/gelu.cpp
    src/ops/attention/layernorm.cpp
    src/ops/attention/softmax.cpp
    src/ops/attention/matmul.cpp
    src/ops/attention/qkv.cpp
    src/ops/attention/attention_scores.cpp
    src/ops/attention/attention_softmax.cpp
    src/ops/attention/scaled_dot_product_attention.cpp

    # Operator Registry
    src/ops/operator_registry.cpp

    # CPU operators
    src/ops/cpu/gelu_operator.cpp
    src/ops/cpu/matmul_operator.cpp
    src/ops/cpu/layernorm_operator.cpp
    src/ops/cpu/attention_scores_operator.cpp
    src/ops/cpu/attention_softmax_operator.cpp

    # CUDA operators
    src/ops/cuda/gelu_operator.cpp
    src/ops/cuda/matmul_operator.cpp
    src/ops/cuda/layernorm_operator.cpp
    src/ops/cuda/attention_scores_operator.cpp
    src/ops/cuda/attention_softmax_operator.cpp
)


target_include_directories(
    cuda_executor_attention_multistream_profile
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(
    cuda_executor_attention_multistream_profile
    PRIVATE
    cuda_backend
    CUDA::cudart
)


set_property(
    TARGET cuda_executor_attention_multistream_profile
    PROPERTY CUDA_ARCHITECTURES 70
)




 
# MatMul Block Configuration Benchmark
add_executable(
    cuda_matmul_block_config_benchmark

    benchmark/cuda/benchmark_matmul_block_config.cu
)


target_include_directories(
    cuda_matmul_block_config_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(
    cuda_matmul_block_config_benchmark
    PRIVATE
    CUDA::cudart
)


set_property(
    TARGET cuda_matmul_block_config_benchmark
    PROPERTY CUDA_ARCHITECTURES 70
)






 
# AttentionScores Block Configuration Benchmark

add_executable(
    cuda_attention_scores_block_config_benchmark

    benchmark/cuda/benchmark_attention_scores_block_config.cu
)


target_include_directories(
    cuda_attention_scores_block_config_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(
    cuda_attention_scores_block_config_benchmark
    PRIVATE
    CUDA::cudart
)


set_property(
    TARGET cuda_attention_scores_block_config_benchmark
    PROPERTY CUDA_ARCHITECTURES 70
)







# AttentionSoftmax Block Size Benchmark
add_executable(
    cuda_attention_softmax_block_config_benchmark

    benchmark/cuda/benchmark_attention_softmax_block_config.cu
)


target_include_directories(
    cuda_attention_softmax_block_config_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)


target_link_libraries(
    cuda_attention_softmax_block_config_benchmark
    PRIVATE
    CUDA::cudart
)


set_property(
    TARGET cuda_attention_softmax_block_config_benchmark
    PROPERTY CUDA_ARCHITECTURES 70
)