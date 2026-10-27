#include "graph/graph.h"
#include "graph/node.h"

#include "tensor/tensor.h"
#include "tensor/pinned_tensor.h"

#include "runtime/executor.h"
#include "runtime/cuda_memory_pool.h"
#include "runtime/pinned_memory_pool.h"
#include "runtime/device_transfer.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


// ==================================================
// One complete Attention request.
//
// Each instance owns:
//
// pinned Q / K / V / output
// CUDA Q / K / V
// Executor
// CUDA stream
// Attention graph
//
// ==================================================

class AttentionPipeline
{
public:

    AttentionPipeline(
        int seq_len,
        int head_dim,
        int seed
    )
        :
        seq_len_(seq_len),
        head_dim_(head_dim),

        q_host_(
            "Q_host",
            {seq_len, head_dim},
            pinned_pool_
        ),

        k_host_(
            "K_host",
            {seq_len, head_dim},
            pinned_pool_
        ),

        v_host_(
            "V_host",
            {seq_len, head_dim},
            pinned_pool_
        ),

        output_host_(
            "output_host",
            {seq_len, head_dim},
            pinned_pool_
        )
    {
        initialize_host_data(
            seed
        );


        allocate_device_inputs();


        build_graph();


        configure_executor();
    }


    AttentionPipeline(
        const AttentionPipeline&
    ) = delete;


    AttentionPipeline& operator=(
        const AttentionPipeline&
    ) = delete;


    ~AttentionPipeline()
    {
        executor_.synchronize();


        if(q_device_ptr_ != nullptr)
        {
            q_device_pool_.release(
                q_device_ptr_
            );

            q_device_ptr_ = nullptr;
        }


        if(k_device_ptr_ != nullptr)
        {
            k_device_pool_.release(
                k_device_ptr_
            );

            k_device_ptr_ = nullptr;
        }


        if(v_device_ptr_ != nullptr)
        {
            v_device_pool_.release(
                v_device_ptr_
            );

            v_device_ptr_ = nullptr;
        }
    }


    // ==================================================
    // Submit one complete Attention pipeline.
    //
    // Same stream:
    //
    // H2D Q
    // H2D K
    // H2D V
    // AttentionScores
    // AttentionSoftmax
    // MatMul
    // D2H output
    //
    // No synchronization here.
    // ==================================================

    void submit()
    {
        copy_host_to_device(
            q_host_,
            *q_device_,
            executor_.get_stream()
        );


        copy_host_to_device(
            k_host_,
            *k_device_,
            executor_.get_stream()
        );


        copy_host_to_device(
            v_host_,
            *v_device_,
            executor_.get_stream()
        );


        executor_.run(
            graph_
        );


        Tensor& device_output =
            executor_.get_tensor(
                "output"
            );


        copy_device_to_host(
            device_output,
            output_host_,
            executor_.get_stream()
        );
    }


    void synchronize()
    {
        executor_.synchronize();
    }


    cudaStream_t stream() const
    {
        return executor_.get_stream();
    }


private:

    int seq_len_;

    int head_dim_;


    // ==================================================
    // Pinned host memory
    // ==================================================

    PinnedMemoryPool pinned_pool_;


    PinnedTensor q_host_;

    PinnedTensor k_host_;

    PinnedTensor v_host_;

    PinnedTensor output_host_;


    // ==================================================
    // CUDA input pools
    // ==================================================

    CudaMemoryPool q_device_pool_;

    CudaMemoryPool k_device_pool_;

    CudaMemoryPool v_device_pool_;


    float* q_device_ptr_ = nullptr;

    float* k_device_ptr_ = nullptr;

    float* v_device_ptr_ = nullptr;


    // ==================================================
    // CUDA Tensor wrappers
    // ==================================================

    std::unique_ptr<Tensor> q_device_;

    std::unique_ptr<Tensor> k_device_;

    std::unique_ptr<Tensor> v_device_;


    // ==================================================
    // Executor + Graph
    // ==================================================

    Executor executor_;

    Graph graph_;


    // ==================================================
    // Input initialization
    // ==================================================

    void initialize_host_data(
        int seed
    )
    {
        const std::size_t size =
            q_host_.size();


        for(std::size_t i = 0; i < size; i++)
        {
            q_host_[i] =
                static_cast<float>(
                    (
                        static_cast<int>(i)
                        + seed
                    )
                    % 31
                    - 15
                )
                * 0.03125f;


            k_host_[i] =
                static_cast<float>(
                    (
                        static_cast<int>(i)
                        + seed * 3
                    )
                    % 37
                    - 18
                )
                * 0.025f;


            v_host_[i] =
                static_cast<float>(
                    (
                        static_cast<int>(i)
                        + seed * 7
                    )
                    % 41
                    - 20
                )
                * 0.02f;
        }
    }


    // ==================================================
    // Allocate CUDA inputs
    // ==================================================

    void allocate_device_inputs()
    {
        q_device_ptr_ =
            q_device_pool_.allocate(
                q_host_.size()
            );


        k_device_ptr_ =
            k_device_pool_.allocate(
                k_host_.size()
            );


        v_device_ptr_ =
            v_device_pool_.allocate(
                v_host_.size()
            );


        q_device_ =
            std::make_unique<Tensor>(
                "Q",
                std::vector<int>{
                    seq_len_,
                    head_dim_
                },
                q_device_ptr_,
                DeviceType::CUDA
            );


        k_device_ =
            std::make_unique<Tensor>(
                "K",
                std::vector<int>{
                    seq_len_,
                    head_dim_
                },
                k_device_ptr_,
                DeviceType::CUDA
            );


        v_device_ =
            std::make_unique<Tensor>(
                "V",
                std::vector<int>{
                    seq_len_,
                    head_dim_
                },
                v_device_ptr_,
                DeviceType::CUDA
            );
    }


    // ==================================================
    // Build Attention graph
    // ==================================================

    void build_graph()
    {
        graph_.add_node(
            Node(
                "attention_scores_node",
                "AttentionScores",
                {"Q", "K"},
                {"scores"}
            )
        );


        graph_.add_node(
            Node(
                "attention_softmax_node",
                "AttentionSoftmax",
                {"scores"},
                {"weights"}
            )
        );


        graph_.add_node(
            Node(
                "attention_output_node",
                "MatMul",
                {"weights", "V"},
                {"output"}
            )
        );


        graph_.build_dependencies();
    }


    // ==================================================
    // Register tensors
    // ==================================================

    void configure_executor()
    {
        executor_.add_tensor(
            *q_device_
        );


        executor_.add_tensor(
            *k_device_
        );


        executor_.add_tensor(
            *v_device_
        );


        executor_.register_tensor(
            "scores",
            {seq_len_, seq_len_},
            DeviceType::CUDA
        );


        executor_.register_tensor(
            "weights",
            {seq_len_, seq_len_},
            DeviceType::CUDA
        );


        executor_.register_tensor(
            "output",
            {seq_len_, head_dim_},
            DeviceType::CUDA
        );
    }
};


// ==================================================
// Warmup
// ==================================================

void warmup(
    AttentionPipeline& a,
    AttentionPipeline& b,
    int runs
)
{
    for(int i = 0; i < runs; i++)
    {
        a.submit();

        b.submit();


        a.synchronize();

        b.synchronize();
    }
}


// ==================================================
// Profile workload
//
// Keep the number of iterations low so that the
// Nsight Systems timeline stays easy to inspect.
// ==================================================

void profile_multistream(
    AttentionPipeline& a,
    AttentionPipeline& b,
    int runs
)
{
    for(int i = 0; i < runs; i++)
    {
        // Submit both complete pipelines first.
        a.submit();

        b.submit();


        // Wait only after both streams
        // already contain work.
        a.synchronize();

        b.synchronize();
    }
}


// ==================================================
// Main
//
// Usage:
//
// ./cuda_executor_attention_multistream_profile 256
//
// ./cuda_executor_attention_multistream_profile 2048
//
// ==================================================

int main(
    int argc,
    char** argv
)
{
    try
    {
        if(argc != 2)
        {
            std::cerr
                << "Usage: "
                << argv[0]
                << " <seq_len>"
                << std::endl;

            return 1;
        }


        const int seq_len =
            std::stoi(
                argv[1]
            );


        if(seq_len <= 0)
        {
            throw std::invalid_argument(
                "seq_len must be positive"
            );
        }


        constexpr int head_dim =
            64;


        constexpr int warmup_runs =
            2;


        // Keep this small for a clean Nsight timeline.
        constexpr int profile_runs =
            3;


        std::cout
            << "Profiling Multi-stream Attention"
            << std::endl;


        std::cout
            << "seq_len: "
            << seq_len
            << std::endl;


        std::cout
            << "head_dim: "
            << head_dim
            << std::endl;


        std::cout
            << "warmup_runs: "
            << warmup_runs
            << std::endl;


        std::cout
            << "profile_runs: "
            << profile_runs
            << std::endl;


        // Two independent requests.
        AttentionPipeline pipeline_a(
            seq_len,
            head_dim,
            1
        );


        AttentionPipeline pipeline_b(
            seq_len,
            head_dim,
            123
        );


        if(
            pipeline_a.stream()
            ==
            pipeline_b.stream()
        )
        {
            throw std::runtime_error(
                "Two pipelines unexpectedly share "
                "the same CUDA stream"
            );
        }


        // ------------------------------
        // Warmup
        // ------------------------------

        warmup(
            pipeline_a,
            pipeline_b,
            warmup_runs
        );


        // Optional full-device sync here makes
        // the profiling region visually cleaner.
        cudaDeviceSynchronize();


        std::cout
            << "Starting profiling region..."
            << std::endl;


        // ------------------------------
        // Profile region
        // ------------------------------

        profile_multistream(
            pipeline_a,
            pipeline_b,
            profile_runs
        );


        cudaDeviceSynchronize();


        std::cout
            << "Profiling region finished."
            << std::endl;


        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr
            << "Profile exception: "
            << e.what()
            << std::endl;

        return 1;
    }
}