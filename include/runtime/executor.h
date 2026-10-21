#ifndef EXECUTOR_H
#define EXECUTOR_H

#include "graph/graph.h"
#include "tensor/tensor.h"

#include "runtime/memory_pool.h"
#include "runtime/memory_planner.h"
#include "runtime/cuda_memory_pool.h"

#include "ops/operator_registry.h"

#include <string>
#include <unordered_map>
#include <cuda_runtime.h>

class Executor{
    
    public:

        Executor();
        ~Executor();

        Executor(const Executor&) = delete;    // Forbid copy constructor.

        Executor& operator =(const Executor&) = delete;  // forbid copy assignment.


        void synchronize();


        void add_tensor(const Tensor& tensor);
        void register_tensor(const std::string& name, const std::vector<int>& shape, DeviceType device = DeviceType::CPU);

        Tensor& get_tensor(const std::string& name);
        const Tensor& get_tensor(const std::string& name) const;

        void run(const Graph& graph);


        // For test, two exposure function
        std::size_t memory_block_count() const;
        std::size_t memory_capacity_count() const;

        // Public access for inner tool
        //OperatorRegistry& operator_registry();
        //const OperatorRegistry& operator_registry() const;


    private:
        std::unordered_map<std::string, Tensor> tensors_;
        std::unordered_map<std::string, std::vector<int>> tensor_shapes_;
        std::unordered_map<std::string, DeviceType> tensor_devices_;
        //OperatorRegistry operator_registry_;
        

        void execute_node(const Node& node);

        MemoryPool memory_pool_;
        CudaMemoryPool cuda_memory_pool_;
        MemoryPlanner memory_planner_;
        void allocate_tensor_if_needed(const std::string& name);
        void release_tensor_if_dead(const std::string& name, int step);

        // sub function of the execute_node.
        /*
        void execute_node_attention_softmax(const Node& node);
        void execute_matmul(const Node& node);
        void execute_gelu(const Node& node);
        void execute_layernorm(const Node& node);
        void execute_attention_scores(const Node& node);
        */
};      


#endif