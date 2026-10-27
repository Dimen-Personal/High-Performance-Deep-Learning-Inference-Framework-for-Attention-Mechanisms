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
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>


// 注意现在multi stream的运用是在提升多个attention运行的时候的吞吐量，比如在运算第一个copy的时候可以开始第二个的allocation。

// ==================================================
// One complete Attention inference pipeline.
//
// Each instance owns:
//
// Pinned Q / K / V / Output
// CUDA Q / K / V
// Executor
// CUDA Stream
// Attention Graph
//
// Pipeline A -> Stream A
// Pipeline B -> Stream B
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


    // Disable copying because this class owns
    // CUDA resources and an Executor.
    AttentionPipeline(
        const AttentionPipeline&
    ) = delete;


    AttentionPipeline& operator=(
        const AttentionPipeline&
    ) = delete;


    ~AttentionPipeline()
    {
        // Make sure no CUDA work is still using
        // the external device input buffers.
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
    // Submit one full inference request.
    //
    // Same stream:
    //
    // H2D Q
    // H2D K
    // H2D V
    //     ↓
    // AttentionScores
    //     ↓
    // AttentionSoftmax
    //     ↓
    // MatMul
    //     ↓
    // D2H Output
    //
    // IMPORTANT:
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


    const PinnedTensor& output() const
    {
        return output_host_;
    }


private:

    // ==================================================
    // Shape
    // ==================================================

    int seq_len_;

    int head_dim_;


    // ==================================================
    // Pinned Host Memory
    //
    // Pool must be declared BEFORE PinnedTensor objects
    // because PinnedTensor uses this pool.
    // ==================================================

    PinnedMemoryPool pinned_pool_;


    PinnedTensor q_host_;

    PinnedTensor k_host_;

    PinnedTensor v_host_;

    PinnedTensor output_host_;


    // ==================================================
    // CUDA Input Memory Pools
    // ==================================================

    CudaMemoryPool q_device_pool_;

    CudaMemoryPool k_device_pool_;

    CudaMemoryPool v_device_pool_;


    // ==================================================
    // Raw CUDA Input Pointers
    // ==================================================

    float* q_device_ptr_ = nullptr;

    float* k_device_ptr_ = nullptr;

    float* v_device_ptr_ = nullptr;


    // ==================================================
    // Device Tensor wrappers
    //
    // Tensor has no default constructor,
    // so use unique_ptr and construct them after
    // CUDA memory has been allocated.
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
    // Initialize pinned host input data.
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
    // Allocate CUDA Q / K / V.
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
    // Build Attention Graph:
    //
    // Q -----\
    //         AttentionScores
    // K -----/
    //              ↓
    //           scores
    //              ↓
    //      AttentionSoftmax
    //              ↓
    //           weights
    //              ↓
    // weights -----\
    //               MatMul
    // V -----------/
    //              ↓
    //            output
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
    // Register tensors with Executor.
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
// Serial Warmup
//
// A:
// H2D -> Attention -> D2H -> synchronize
//
// B:
// H2D -> Attention -> D2H -> synchronize
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


// ==================================================
// Multi-stream Warmup
//
// Submit both requests before waiting.
//
// Stream A:
// H2D A -> Attention A -> D2H A
//
// Stream B:
// H2D B -> Attention B -> D2H B
// ==================================================

void warmup_multistream(
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
// Serial benchmark.
//
// Measures the total end-to-end time
// required to finish TWO requests.
//
// Request A completely finishes first,
// then Request B starts.
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
        // Request A
        a.submit();

        a.synchronize();


        // Request B
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


    return elapsed_ms
        /
        static_cast<double>(
            runs
        );
}


// ==================================================
// Multi-stream benchmark.
//
// Measures total time required to finish
// TWO requests.
//
// Both requests are submitted before either
// one is synchronized.
//
// CUDA may overlap:
// - copy and compute
// - kernels from independent streams
//
// depending on available GPU resources.
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
        // Submit complete pipeline A.
        a.submit();


        // Immediately submit complete pipeline B.
        b.submit();


        // Only wait after both streams
        // already contain their work.
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


    return elapsed_ms
        /
        static_cast<double>(
            runs
        );
}


// ==================================================
// Run one problem size.
// ==================================================

void run_benchmark(
    int seq_len,
    int head_dim,
    int warmup_runs,
    int benchmark_runs
)
{
    // Different seeds guarantee that the two
    // requests contain different input values.
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


    // ==================================================
    // Sanity check:
    // Two Executors must have two different streams.
    // ==================================================

    if(
        pipeline_a.stream()
        ==
        pipeline_b.stream()
    )
    {
        throw std::runtime_error(
            "Two Attention pipelines unexpectedly "
            "share the same CUDA stream"
        );
    }


    // ==================================================
    // Warmup
    // ==================================================

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


    // ==================================================
    // Serial
    // ==================================================

    const double serial_ms =
        benchmark_serial(
            pipeline_a,
            pipeline_b,
            benchmark_runs
        );


    // ==================================================
    // Multi-stream
    // ==================================================

    const double multistream_ms =
        benchmark_multistream(
            pipeline_a,
            pipeline_b,
            benchmark_runs
        );


    // ==================================================
    // Speedup
    //
    // > 1.0 = multi-stream faster
    // = 1.0 = approximately equal
    // < 1.0 = multi-stream slower
    // ==================================================

    const double speedup =
        serial_ms
        /
        multistream_ms;


    // ==================================================
    // Throughput
    //
    // Each benchmark iteration processes
    // TWO requests.
    //
    // serial_ms is milliseconds / 2 requests.
    //
    // Requests / second:
    //
    // 2 * 1000 / milliseconds
    // ==================================================

    const double serial_requests_per_second =
        2000.0
        /
        serial_ms;


    const double multistream_requests_per_second =
        2000.0
        /
        multistream_ms;


    // ==================================================
    // Print one row.
    // ==================================================

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
        // Standard transformer-like head dimension.
        constexpr int head_dim =
            64;


        // Warmup avoids measuring:
        //
        // first CUDA context initialization,
        // initial allocations,
        // one-time runtime overhead.
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
            << "\n"
            << "End-to-End Attention Multi-stream Benchmark"
            << "\n"
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


        for(
            int seq_len :
            sequence_lengths
        )
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