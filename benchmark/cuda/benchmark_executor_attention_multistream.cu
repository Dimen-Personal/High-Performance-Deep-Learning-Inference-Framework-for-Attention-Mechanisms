#include "graph/graph.h"
#include "graph/node.h"

#include "tensor/tensor.h"
#include "tensor/pinned_tensor.h"

#include "runtime/executor.h"
#include "runtime/cuda_memory_pool.h"
#include "runtime/pinned_memory_pool.h"
#include "runtime/device_transfer.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


// CUDA error checking
void check_cuda(
    cudaError_t status,
    const char* operation
){
    if(status != cudaSuccess)
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
}



// ==================================================
// One complete Attention inference pipeline.
//
// Each instance owns:
//
// Pinned Q/K/V/output
// CUDA Q/K/V
// Executor
// CUDA stream
// Attention graph
//
// Therefore:
//
// Pipeline A -> Stream A
// Pipeline B -> Stream B
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
        // Make sure the stream no longer uses
        // external CUDA input buffers.
        executor_.synchronize();


        q_device_pool_.release(
            q_device_ptr_
        );


        k_device_pool_.release(
            k_device_ptr_
        );


        v_device_pool_.release(
            v_device_ptr_
        );
    }


    // ==================================================
    // Submit:
    //
    // Q H2D
    // K H2D
    // V H2D
    // AttentionScores
    // AttentionSoftmax
    // MatMul
    // Output D2H
    //
    // No synchronization here.
    // ==================================================

    void submit()
    {
        copy_host_to_device(
            q_host_,
            q_device_,
            executor_.get_stream()
        );


        copy_host_to_device(
            k_host_,
            k_device_,
            executor_.get_stream()
        );


        copy_host_to_device(
            v_host_,
            v_device_,
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


    const PinnedTensor& output() const
    {
        return output_host_;
    }


private:

    int seq_len_;
    int head_dim_;


    // ==================================================
    // Host pinned memory
    // ==================================================

    PinnedMemoryPool pinned_pool_;


    PinnedTensor q_host_;
    PinnedTensor k_host_;
    PinnedTensor v_host_;

    PinnedTensor output_host_;


    // ==================================================
    // CUDA input memory
    // ==================================================

    CudaMemoryPool q_device_pool_;
    CudaMemoryPool k_device_pool_;
    CudaMemoryPool v_device_pool_;


    float* q_device_ptr_ = nullptr;
    float* k_device_ptr_ = nullptr;
    float* v_device_ptr_ = nullptr;


    // These Tensor objects do not own memory.
    Tensor q_device_;

    Tensor k_device_;

    Tensor v_device_;


    // ==================================================
    // Executor + Graph
    // ==================================================

    Executor executor_;

    Graph graph_;


    // ==================================================
    // Host data initialization
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
    // Allocate GPU Q/K/V.
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
            Tensor(
                "Q",
                {seq_len_, head_dim_},
                q_device_ptr_,
                DeviceType::CUDA
            );


        k_device_ =
            Tensor(
                "K",
                {seq_len_, head_dim_},
                k_device_ptr_,
                DeviceType::CUDA
            );


        v_device_ =
            Tensor(
                "V",
                {seq_len_, head_dim_},
                v_device_ptr_,
                DeviceType::CUDA
            );
    }


    // ==================================================
    // Build:
    //
    // Q,K
    //  ↓
    // AttentionScores
    //  ↓
    // scores
    //  ↓
    // AttentionSoftmax
    //  ↓
    // weights
    //  ↓
    // MatMul(weights,V)
    //  ↓
    // output
    //
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
    // Configure Executor.
    // ==================================================

    void configure_executor()
    {
        executor_.add_tensor(
            q_device_
        );


        executor_.add_tensor(
            k_device_
        );


        executor_.add_tensor(
            v_device_
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

void warmup_serial(
    AttentionPipeline& a,
    AttentionPipeline& b,
    int runs
)
{
    for(int i = 0; i < runs; i++)
    {
        a.submit();

        a.synchronize();


        b.submit();

        b.synchronize();
    }
}


void warmup_multistream(
    AttentionPipeline& a,
    AttentionPipeline& b,
    int runs
)
{
    for(int i = 0; i < runs; i++)
    {
        // Submit both before waiting.
        a.submit();

        b.submit();


        a.synchronize();

        b.synchronize();
    }
}


// ==================================================
// Serial benchmark.
//
// Request A:
//
// H2D -> Attention -> D2H -> wait
//
// then:
//
// Request B:
//
// H2D -> Attention -> D2H -> wait
//
// ==================================================

double benchmark_serial(
    AttentionPipeline& a,
    AttentionPipeline& b,
    int runs
)
{
    using Clock =
        std::chrono::steady_clock;


    const auto start =
        Clock::now();


    for(int i = 0; i < runs; i++)
    {
        a.submit();

        a.synchronize();


        b.submit();

        b.synchronize();
    }


    const auto end =
        Clock::now();


    const double elapsed_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            end - start
        ).count();


    return elapsed_ms / runs;
}


// ==================================================
// Multi-stream benchmark.
//
// Submit:
//
// Stream A:
// H2D A -> Attention A -> D2H A
//
// Stream B:
// H2D B -> Attention B -> D2H B
//
// Then wait.
//
// ==================================================

double benchmark_multistream(
    AttentionPipeline& a,
    AttentionPipeline& b,
    int runs
)
{
    using Clock =
        std::chrono::steady_clock;


    const auto start =
        Clock::now();


    for(int i = 0; i < runs; i++)
    {
        // IMPORTANT:
        // submit both before synchronization.

        a.submit();

        b.submit();


        a.synchronize();

        b.synchronize();
    }


    const auto end =
        Clock::now();


    const double elapsed_ms =
        std::chrono::duration<
            double,
            std::milli
        >(
            end - start
        ).count();


    return elapsed_ms / runs;
}


// ==================================================
// Benchmark one problem size.
// ==================================================

void run_benchmark(
    int seq_len,
    int head_dim,
    int warmup_runs,
    int benchmark_runs
)
{
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


    // Verify that we really have two streams.
    if(
        pipeline_a.stream()
        ==
        pipeline_b.stream()
    )
    {
        throw std::runtime_error(
            "Two Attention pipelines share the same CUDA stream"
        );
    }


    // ------------------------------
    // Warmup
    // ------------------------------

    warmup_serial(
        pipeline_a,
        pipeline_b,
        warmup_runs
    );


    warmup_multistream(
        pipeline_a,
        pipeline_b,
        warmup_runs
    );


    // ------------------------------
    // Benchmark serial.
    // ------------------------------

    const double serial_ms =
        benchmark_serial(
            pipeline_a,
            pipeline_b,
            benchmark_runs
        );


    // ------------------------------
    // Benchmark multi-stream.
    // ------------------------------

    const double multistream_ms =
        benchmark_multistream(
            pipeline_a,
            pipeline_b,
            benchmark_runs
        );


    const double speedup =
        serial_ms
        /
        multistream_ms;


    // Total latency is for TWO requests.
    //
    // Therefore requests per second:
    //
    // 2 requests / iteration.
    // ------------------------------

    const double serial_requests_per_second =
        2000.0
        /
        serial_ms;


    const double multistream_requests_per_second =
        2000.0
        /
        multistream_ms;


    std::cout
        << std::left
        << std::setw(10)
        << seq_len

        << std::setw(12)
        << head_dim

        << std::setw(18)
        << std::fixed
        << std::setprecision(4)
        << serial_ms

        << std::setw(18)
        << multistream_ms

        << std::setw(14)
        << speedup

        << std::setw(18)
        << serial_requests_per_second

        << std::setw(18)
        << multistream_requests_per_second

        << std::endl;
}


// ==================================================
// Main
// ==================================================

int main()
{
    try
    {
        constexpr int head_dim =
            64;


        constexpr int warmup_runs =
            3;


        constexpr int benchmark_runs =
            10;


        const std::vector<int> sequence_lengths =
        {
            128,
            256,
            512,
            1024,
            2048
        };


        std::cout
            << "\nEnd-to-End Attention Multi-stream Benchmark\n"
            << std::endl;


        std::cout
            << std::left

            << std::setw(10)
            << "SeqLen"

            << std::setw(12)
            << "HeadDim"

            << std::setw(18)
            << "Serial(ms)"

            << std::setw(18)
            << "Multi(ms)"

            << std::setw(14)
            << "Speedup"

            << std::setw(18)
            << "Serial req/s"

            << std::setw(18)
            << "Multi req/s"

            << std::endl;


        std::cout
            << std::string(
                108,
                '-'
            )
            << std::endl;


        for(int seq_len : sequence_lengths)
        {
            run_benchmark(
                seq_len,
                head_dim,
                warmup_runs,
                benchmark_runs
            );
        }


        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr
            << "Benchmark exception: "
            << e.what()
            << std::endl;


        return 1;
    }
}