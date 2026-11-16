#include "cpu/fused_matmul_bias_gelu.h"

#include <cstddef>
#include <iostream>
#include <vector>


constexpr int M = 512;
constexpr int K = 512;
constexpr int N = 512;

constexpr int WARMUP_RUNS = 0;
constexpr int PROFILE_RUNS = 1;


// Prevent the compiler from considering the result unused.
volatile float result_sink = 0.0f;


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
            static_cast<float>(value)
            *
            0.01f;
    }
}


int main()
{
    const std::size_t a_elements =
        static_cast<std::size_t>(M)
        *
        K;


    const std::size_t b_elements =
        static_cast<std::size_t>(K)
        *
        N;


    const std::size_t output_elements =
        static_cast<std::size_t>(M)
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


    std::vector<float> output(
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


    std::cout
        << "Fused MatMul + Bias + GELU Profile"
        << std::endl;


    std::cout
        << "M="
        << M
        << ", K="
        << K
        << ", N="
        << N
        << std::endl;


    // ========================================================
    // Warmup
    // ========================================================

    for(
        int i = 0;
        i < WARMUP_RUNS;
        i++
    )
    {
        fused_matmul_bias_gelu(
            A.data(),
            B.data(),
            bias.data(),
            output.data(),
            M,
            K,
            N
        );
    }


    // ========================================================
    // Profile Runs
    // ========================================================

    for(
        int i = 0;
        i < PROFILE_RUNS;
        i++
    )
    {
        fused_matmul_bias_gelu(
            A.data(),
            B.data(),
            bias.data(),
            output.data(),
            M,
            K,
            N
        );


        result_sink +=
            output[
                static_cast<std::size_t>(
                    i
                )
                %
                output.size()
            ];
    }


    std::cout
        << "Result sink: "
        << result_sink
        << std::endl;


    return 0;
}