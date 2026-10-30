#include "quantization/quantization.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>


struct ErrorMetrics{
    float mae = 0.0f;

    float max_error = 0.0f;
};


ErrorMetrics calculate_error(
    const std::vector<float>& reference,
    const std::vector<float>& reconstructed
){
    if(reference.size() != reconstructed.size())
        throw std::runtime_error("Tensor sizes do not match.");

    
    ErrorMetrics metrics;

    if(reference.empty())
        return metrics;

    double total_error = 0.0;




    for(std::size_t i = 0; i < reference.size(); i++){
        const float error = std::fabs(reference[i] - reconstructed[i]);

        total_error += static_cast<double> (error);
        metrics.max_error = std::max(metrics.max_error, error);
    }



    metrics.mae = static_cast<float> (total_error / static_cast<double>(reference.size()));


    return metrics;
}






int main(){
    try{
        const std::vector<float> input =
        {
            -2.0f,
            -1.5f,
            -1.0f,
            -0.5f,
             0.0f,
             0.25f,
             0.5f,
             1.0f,
             1.5f,
             2.0f
        };


        const QuantizedTensor quantized = quantize_fp32_to_int8(input.data(), input.size());


        const std::vector<float> reconstructed = dequantize_int8_to_fp32(
            quantized.data.data(),
            quantized.data.size(),
            quantized.params
        );




        // BIG Print (by GPT)

        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "INT8 Symmetric Quantization Test"
            << std::endl;


        std::cout
            << "============================================"
            << std::endl;


        std::cout
            << "Scale: "
            << std::fixed
            << std::setprecision(8)
            << quantized.params.scale
            << std::endl;


        std::cout
            << "Zero Point: "
            << quantized.params.zero_point
            << std::endl;


        std::cout
            << std::endl;


        // ====================================================
        // Print values
        // ====================================================

        std::cout
            << std::left

            << std::setw(16)
            << "FP32"

            << std::setw(16)
            << "INT8"

            << std::setw(16)
            << "Dequantized"

            << std::setw(16)
            << "Abs Error"

            << std::endl;


        std::cout
            << std::string(
                64,
                '-'
            )
            << std::endl;


        for(
            std::size_t i = 0;
            i < input.size();
            i++
        )
        {
            const float error =
                std::fabs(
                    input[i]
                    -
                    reconstructed[i]
                );


            std::cout
                << std::left

                << std::setw(16)
                << std::fixed
                << std::setprecision(6)
                << input[i]

                << std::setw(16)
                << static_cast<int>(
                    quantized.data[i]
                )

                << std::setw(16)
                << std::fixed
                << std::setprecision(6)
                << reconstructed[i]

                << std::setw(16)
                << std::fixed
                << std::setprecision(6)
                << error

                << std::endl;
        }



        // ====================================================
        // Error metrics
        // ====================================================

        const ErrorMetrics metrics =
            calculate_error(
                input,
                reconstructed
            );


        std::cout
            << "\n============================================"
            << std::endl;


        std::cout
            << "Error Metrics"
            << std::endl;


        std::cout
            << "============================================"
            << std::endl;


        std::cout
            << "MAE: "
            << metrics.mae
            << std::endl;


        std::cout
            << "Max Absolute Error: "
            << metrics.max_error
            << std::endl;


        // ====================================================
        // Theoretical maximum quantization error
        //
        // Rounding to nearest quantization point should
        // normally produce error <= scale / 2.
        // ====================================================

        const float theoretical_max_error =
            quantized.params.scale
            /
            2.0f;


        std::cout
            << "Expected rounding error bound: ~"
            << theoretical_max_error
            << std::endl;


        // ====================================================
        // Correctness check
        //
        // Allow a tiny floating-point margin.
        // ====================================================

        const float tolerance =
            theoretical_max_error
            +
            1e-5f;


        if(
            metrics.max_error
            >
            tolerance
        )
        {
            std::cerr
                << "\nQuantization test FAILED."
                << std::endl;


            return 1;
        }


        // ====================================================
        // Verify extreme values
        // ====================================================

        if(
            quantized.data.front()
            !=
            -127
        )
        {
            std::cerr
                << "Negative maximum did not map to -127."
                << std::endl;


            return 1;
        }


        if(
            quantized.data.back()
            !=
            127
        )
        {
            std::cerr
                << "Positive maximum did not map to 127."
                << std::endl;


            return 1;
        }


        std::cout
            << "\nQuantization test PASSED."
            << std::endl;


        return 0;



    } catch (const std::exception& exception){
        std::cerr << "Test Exeception: " << exception.what() << std::endl;

        return 1;
    }
}