#include "cpu/fused_matmul_bias_gelu.h"
#include "cpu/cache_aware_attention.h"
#include "ops/operator.h"

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>


// ============================================================
// Benchmark Configuration
// ============================================================

constexpr int WARMUP_RUNS =
    3;


constexpr int BENCHMARK_RUNS =
    10;


// ============================================================
// Stage A Result
// ============================================================

struct FusionBenchmarkResult
{
    int M =
        0;

    int K =
        0;

    int N =
        0;


    double unfused_ms =
        0.0;


    double fused_ms =
        0.0;


    double speedup =
        0.0;


    double unfused_temporary_memory_mb =
        0.0;


    double fused_temporary_memory_mb =
        0.0;


    double unfused_intermediate_traffic_mb =
        0.0;


    double fused_intermediate_traffic_mb =
        0.0;
};


// ============================================================
// Stage B Result
// ============================================================

struct AttentionBenchmarkResult
{
    int seq_len =
        0;


    double baseline_ms =
        0.0;


    double tiled_ms =
        0.0;


    double speedup =
        0.0;


    double baseline_temp_mb =
        0.0;


    double tiled_working_set_kb =
        0.0;
};


// ============================================================
// Deterministic Initialization
// ============================================================

void initialize_input(
    std::vector<float>& data,
    int seed
)
{
    for(
        std::size_t i = 0;
        i < data.size();
        i++
    )
    {
        const int value =
            (
                static_cast<int>(i)
                +
                seed
            )
            %
            101
            -
            50;


        data[i] =
            static_cast<float>(
                value
            )
            *
            0.01f;
    }
}


// ============================================================
// Generic Benchmark Helper
// ============================================================

template<typename Function>
double benchmark_function(
    Function function
)
{
    // ========================================================
    // Warmup
    // ========================================================

    for(
        int i = 0;
        i < WARMUP_RUNS;
        i++
    )
    {
        function();
    }


    // ========================================================
    // Timed Runs
    // ========================================================

    const auto start =
        std::chrono::high_resolution_clock::now();


    for(
        int i = 0;
        i < BENCHMARK_RUNS;
        i++
    )
    {
        function();
    }


    const auto end =
        std::chrono::high_resolution_clock::now();


    const std::chrono::duration<double, std::milli>
        elapsed =
            end
            -
            start;


    return
        elapsed.count()
        /
        static_cast<double>(
            BENCHMARK_RUNS
        );
}


// ============================================================
// Stage A
// Fusion Benchmark
// ============================================================

FusionBenchmarkResult run_fusion_shape(
    int M,
    int K,
    int N
)
{
    const std::size_t a_elements =
        static_cast<std::size_t>(
            M
        )
        *
        K;


    const std::size_t b_elements =
        static_cast<std::size_t>(
            K
        )
        *
        N;


    const std::size_t output_elements =
        static_cast<std::size_t>(
            M
        )
        *
        N;


    std::vector<float> A(
        a_elements
    );


    std::vector<float> B(
        b_elements
    );


    std::vector<float> bias(
        N
    );


    std::vector<float> temp_matmul(
        output_elements
    );


    std::vector<float> temp_bias(
        output_elements
    );


    std::vector<float> unfused_output(
        output_elements
    );


    std::vector<float> fused_output(
        output_elements
    );


    initialize_input(
        A,
        1
    );


    initialize_input(
        B,
        17
    );


    initialize_input(
        bias,
        31
    );


    // ========================================================
    // Unfused
    // ========================================================

    const double unfused_ms =
        benchmark_function(
            [&]()
            {
                unfused_matmul_bias_gelu(
                    A.data(),
                    B.data(),
                    bias.data(),
                    temp_matmul.data(),
                    temp_bias.data(),
                    unfused_output.data(),
                    M,
                    K,
                    N
                );
            }
        );


    // ========================================================
    // Fused
    // ========================================================

    const double fused_ms =
        benchmark_function(
            [&]()
            {
                fused_matmul_bias_gelu(
                    A.data(),
                    B.data(),
                    bias.data(),
                    fused_output.data(),
                    M,
                    K,
                    N
                );
            }
        );


    const double output_bytes =
        static_cast<double>(
            output_elements
        )
        *
        sizeof(float);


    const double unfused_temporary_bytes =
        2.0
        *
        output_bytes;


    const double fused_temporary_bytes =
        0.0;


    const double unfused_traffic_bytes =
        5.0
        *
        output_bytes;


    const double fused_traffic_bytes =
        1.0
        *
        output_bytes;


    FusionBenchmarkResult result;


    result.M =
        M;


    result.K =
        K;


    result.N =
        N;


    result.unfused_ms =
        unfused_ms;


    result.fused_ms =
        fused_ms;


    result.speedup =
        unfused_ms
        /
        fused_ms;


    result.unfused_temporary_memory_mb =
        unfused_temporary_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.fused_temporary_memory_mb =
        fused_temporary_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.unfused_intermediate_traffic_mb =
        unfused_traffic_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.fused_intermediate_traffic_mb =
        fused_traffic_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    return result;
}


// ============================================================
// Stage B
// Cache-Aware Attention Benchmark
// ============================================================

AttentionBenchmarkResult run_attention_shape(
    int seq_len,
    int head_dim
)
{
    const std::size_t elements =
        static_cast<std::size_t>(
            seq_len
        )
        *
        head_dim;


    std::vector<float> Q(
        elements
    );


    std::vector<float> K(
        elements
    );


    std::vector<float> V(
        elements
    );


    std::vector<float> baseline_output(
        elements
    );


    std::vector<float> tiled_output(
        elements
    );


    initialize_input(
        Q,
        1
    );


    initialize_input(
        K,
        17
    );


    initialize_input(
        V,
        37
    );


    // ========================================================
    // Baseline
    // ========================================================

    const double baseline_ms =
        benchmark_function(
            [&]()
            {
                scaled_dot_product_attention(
                    Q.data(),
                    K.data(),
                    V.data(),
                    baseline_output.data(),
                    seq_len,
                    head_dim
                );
            }
        );


    // ========================================================
    // Cache-Aware Tiled
    // ========================================================

    constexpr int QUERY_BLOCK_SIZE =
        16;


    constexpr int KEY_BLOCK_SIZE =
        32;


    const double tiled_ms =
        benchmark_function(
            [&]()
            {
                cache_aware_attention_tiled(
                    Q.data(),
                    K.data(),
                    V.data(),
                    tiled_output.data(),
                    seq_len,
                    head_dim,
                    QUERY_BLOCK_SIZE,
                    KEY_BLOCK_SIZE
                );
            }
        );


    // ========================================================
    // Baseline Temporary Memory
    //
    // scores[N,N]
    // weights[N,N]
    // ========================================================

    const double baseline_temp_bytes =
        2.0
        *
        static_cast<double>(
            seq_len
        )
        *
        static_cast<double>(
            seq_len
        )
        *
        sizeof(float);


    // ========================================================
    // Tiled Working Set
    //
    // q_tile
    // k_tile
    // v_tile
    // score_tile
    // row_max
    // row_sum
    // output_accumulator
    // ========================================================

    const double tiled_floats =
        static_cast<double>(
            QUERY_BLOCK_SIZE
            *
            head_dim
        )
        +
        static_cast<double>(
            KEY_BLOCK_SIZE
            *
            head_dim
        )
        +
        static_cast<double>(
            KEY_BLOCK_SIZE
            *
            head_dim
        )
        +
        static_cast<double>(
            QUERY_BLOCK_SIZE
            *
            KEY_BLOCK_SIZE
        )
        +
        QUERY_BLOCK_SIZE
        +
        QUERY_BLOCK_SIZE
        +
        static_cast<double>(
            QUERY_BLOCK_SIZE
            *
            head_dim
        );


    const double tiled_bytes =
        tiled_floats
        *
        sizeof(float);


    AttentionBenchmarkResult result;


    result.seq_len =
        seq_len;


    result.baseline_ms =
        baseline_ms;


    result.tiled_ms =
        tiled_ms;


    result.speedup =
        baseline_ms
        /
        tiled_ms;


    result.baseline_temp_mb =
        baseline_temp_bytes
        /
        (
            1024.0
            *
            1024.0
        );


    result.tiled_working_set_kb =
        tiled_bytes
        /
        1024.0;


    return result;
}


// ============================================================
// Print Stage A Performance
// ============================================================

void print_fusion_performance(
    const std::vector<FusionBenchmarkResult>& results
)
{
    std::cout
        << "\n============================================================"
        << std::endl;


    std::cout
        << "Stage A - CPU MatMul + Bias + GELU Fusion Benchmark"
        << std::endl;


    std::cout
        << "============================================================"
        << std::endl;


    std::cout
        << std::left

        << std::setw(8)
        << "M"

        << std::setw(8)
        << "K"

        << std::setw(8)
        << "N"

        << std::setw(18)
        << "Unfused(ms)"

        << std::setw(18)
        << "Fused(ms)"

        << std::setw(16)
        << "Speedup"

        << std::endl;


    std::cout
        << std::string(
            76,
            '-'
        )
        << std::endl;


    for(
        const FusionBenchmarkResult& result :
        results
    )
    {
        std::cout
            << std::left

            << std::setw(8)
            << result.M

            << std::setw(8)
            << result.K

            << std::setw(8)
            << result.N

            << std::setw(18)
            << std::fixed
            << std::setprecision(6)
            << result.unfused_ms

            << std::setw(18)
            << result.fused_ms

            << std::setw(16)
            << std::setprecision(4)
            << result.speedup

            << std::endl;
    }
}


// ============================================================
// Print Stage A Memory
// ============================================================

void print_fusion_memory(
    const std::vector<FusionBenchmarkResult>& results
)
{
    std::cout
        << "\nStage A - Intermediate Memory / Traffic"
        << std::endl;


    std::cout
        << std::left

        << std::setw(8)
        << "M"

        << std::setw(8)
        << "K"

        << std::setw(8)
        << "N"

        << std::setw(22)
        << "UnfusedTemp(MB)"

        << std::setw(20)
        << "FusedTemp(MB)"

        << std::setw(24)
        << "UnfusedTraffic(MB)"

        << std::setw(22)
        << "FusedTraffic(MB)"

        << std::endl;


    std::cout
        << std::string(
            112,
            '-'
        )
        << std::endl;


    for(
        const FusionBenchmarkResult& result :
        results
    )
    {
        std::cout
            << std::left

            << std::setw(8)
            << result.M

            << std::setw(8)
            << result.K

            << std::setw(8)
            << result.N

            << std::setw(22)
            << std::fixed
            << std::setprecision(4)
            << result.unfused_temporary_memory_mb

            << std::setw(20)
            << result.fused_temporary_memory_mb

            << std::setw(24)
            << result.unfused_intermediate_traffic_mb

            << std::setw(22)
            << result.fused_intermediate_traffic_mb

            << std::endl;
    }
}


// ============================================================
// Print Stage B
// ============================================================

void print_attention_results(
    const std::vector<AttentionBenchmarkResult>& results
)
{
    std::cout
        << "\n============================================================"
        << std::endl;


    std::cout
        << "Stage B - CPU Cache-Aware Attention Benchmark"
        << std::endl;


    std::cout
        << "============================================================"
        << std::endl;


    std::cout
        << "Query Block Size: 16"
        << std::endl;


    std::cout
        << "Key Block Size: 32"
        << std::endl;


    std::cout
        << "Head Dimension: 64"
        << std::endl;


    std::cout
        << std::endl;


    std::cout
        << std::left

        << std::setw(12)
        << "SeqLen"

        << std::setw(18)
        << "Baseline(ms)"

        << std::setw(18)
        << "Tiled(ms)"

        << std::setw(16)
        << "Speedup"

        << std::setw(22)
        << "BaseTemp(MB)"

        << std::setw(22)
        << "TileWorking(KB)"

        << std::endl;


    std::cout
        << std::string(
            108,
            '-'
        )
        << std::endl;


    for(
        const AttentionBenchmarkResult& result :
        results
    )
    {
        std::cout
            << std::left

            << std::setw(12)
            << result.seq_len

            << std::setw(18)
            << std::fixed
            << std::setprecision(6)
            << result.baseline_ms

            << std::setw(18)
            << result.tiled_ms

            << std::setw(16)
            << std::setprecision(4)
            << result.speedup

            << std::setw(22)
            << std::setprecision(4)
            << result.baseline_temp_mb

            << std::setw(22)
            << std::setprecision(4)
            << result.tiled_working_set_kb

            << std::endl;
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        std::cout
            << "\n============================================================"
            << std::endl;


        std::cout
            << "Phase 5 Alternative CPU Optimization Benchmark"
            << std::endl;


        std::cout
            << "============================================================"
            << std::endl;


        std::cout
            << "Warmup Runs: "
            << WARMUP_RUNS
            << std::endl;


        std::cout
            << "Benchmark Runs: "
            << BENCHMARK_RUNS
            << std::endl;


        // ====================================================
        // Stage A
        // ====================================================

        struct FusionShape
        {
            int M;
            int K;
            int N;
        };


        const std::vector<FusionShape>
            fusion_shapes =
        {
            {32,   128, 128},
            {64,   256, 256},
            {128,  256, 512},
            {256,  512, 512},
            {512,  512, 512},
            {768,  512, 768},
            {1024, 512, 1024}
        };


        std::vector<FusionBenchmarkResult>
            fusion_results;


        for(
            const FusionShape& shape :
            fusion_shapes
        )
        {
            std::cout
                << "Running Fusion M="
                << shape.M
                << ", K="
                << shape.K
                << ", N="
                << shape.N
                << "..."
                << std::endl;


            fusion_results.push_back(
                run_fusion_shape(
                    shape.M,
                    shape.K,
                    shape.N
                )
            );
        }


        print_fusion_performance(
            fusion_results
        );


        print_fusion_memory(
            fusion_results
        );


        // ====================================================
        // Stage B
        // ====================================================

        constexpr int ATTENTION_HEAD_DIM =
            64;


        const std::vector<int>
            attention_sequence_lengths =
        {
            128,
            256,
            512,
            1024
        };


        std::vector<AttentionBenchmarkResult>
            attention_results;


        for(
            int seq_len :
            attention_sequence_lengths
        )
        {
            std::cout
                << "Running Attention seq_len="
                << seq_len
                << "..."
                << std::endl;


            attention_results.push_back(
                run_attention_shape(
                    seq_len,
                    ATTENTION_HEAD_DIM
                )
            );
        }


        print_attention_results(
            attention_results
        );


        // ====================================================
        // Notes
        // ====================================================

        std::cout
            << "\nNotes:"
            << std::endl;


        std::cout
            << "Stage A speedup = UnfusedTime / FusedTime."
            << std::endl;


        std::cout
            << "Stage B speedup = BaselineAttentionTime / TiledAttentionTime."
            << std::endl;


        std::cout
            << "Speedup > 1.0 means the optimized version is faster."
            << std::endl;


        std::cout
            << "Stage A traffic values are logical intermediate/output traffic estimates."
            << std::endl;


        std::cout
            << "Stage B baseline temporary memory counts scores[N,N] + weights[N,N]."
            << std::endl;


        std::cout
            << "Stage B tiled working-set memory is approximately independent of sequence length."
            << std::endl;


        return 0;
    }
    catch(
        const std::exception& exception
    )
    {
        std::cerr
            << "Benchmark exception: "
            << exception.what()
            << std::endl;


        return 1;
    }
}