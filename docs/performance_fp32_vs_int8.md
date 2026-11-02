# FP32 vs INT8 MatMul Performance Report

## 1. Objective

This experiment compares the performance, memory footprint, and numerical behavior of the current FP32 MatMul implementation against the newly implemented INT8 MatMul path.

The INT8 pipeline is:

```text
FP32 input
→ per-tensor symmetric INT8 quantization
→ INT8 × INT8
→ INT32 accumulation
→ FP32 output
```

The benchmark separately measures:

1. FP32 MatMul latency.
2. INT8 MatMul kernel-only latency.
3. INT8 end-to-end latency including runtime quantization.
4. Numerical error relative to FP32.
5. Raw input-memory reduction.

The goal is to establish a clean baseline before adding OpenMP, cache blocking, SIMD, or architecture-specific INT8 instructions.

---

## 2. Experimental Setup

```text
Warmup Runs: 3
Benchmark Runs: 10
```

Tested square matrix sizes:

```text
128 × 128
256 × 256
512 × 512
1024 × 1024
```

Speedup definitions:

```text
KernelSpeedup = FP32 time / INT8 kernel-only time

E2ESpeedup = FP32 time / (quantization + INT8 MatMul)
```

Interpretation:

```text
Speedup > 1.0  → INT8 is faster
Speedup = 1.0  → approximately equal
Speedup < 1.0  → INT8 is slower
```

---

## 3. Performance Results

| Matrix Size | FP32 (ms) | INT8 Kernel (ms) | INT8 E2E (ms) | Kernel Speedup | E2E Speedup | MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5.9154 | 7.6955 | 7.9469 | 0.7687× | 0.7444× | 0.016874 |
| 256 | 47.3609 | 65.4724 | 67.9806 | 0.7234× | 0.6967× | 0.023688 |
| 512 | 693.5223 | 526.0128 | 533.7385 | 1.3185× | 1.2994× | 0.033505 |
| 1024 | 6794.1257 | 6250.8679 | 6313.0029 | 1.0869× | 1.0762× | 0.047411 |

---

## 4. Memory Results

| Matrix Size | FP32 Input (MB) | INT8 Input (MB) | Reduction |
|---:|---:|---:|---:|
| 128 | 0.1250 | 0.0312 | 75.00% |
| 256 | 0.5000 | 0.1250 | 75.00% |
| 512 | 2.0000 | 0.5000 | 75.00% |
| 1024 | 8.0000 | 2.0000 | 75.00% |

The raw input storage reduction is exactly:

```text
75%
```

because FP32 uses 4 bytes per element while INT8 uses 1 byte per element.

---

## 5. Matrix Size 128

Measured times:

```text
FP32:
5.9154 ms

INT8 kernel-only:
7.6955 ms

INT8 end-to-end:
7.9469 ms
```

Kernel speedup:

```text
0.7687×
```

End-to-end speedup:

```text
0.7444×
```

The INT8 kernel is approximately:

```text
7.6955 / 5.9154 ≈ 1.301×
```

or about:

```text
30.1% slower
```

than FP32.

Including quantization, the end-to-end path is approximately:

```text
34.3% slower
```

than FP32.

---

## 6. Matrix Size 256

Measured times:

```text
FP32:
47.3609 ms

INT8 kernel-only:
65.4724 ms

INT8 end-to-end:
67.9806 ms
```

Kernel speedup:

```text
0.7234×
```

End-to-end speedup:

```text
0.6967×
```

The INT8 kernel is approximately:

```text
38.2% slower
```

than FP32.

Including runtime quantization, the full INT8 path is approximately:

```text
43.5% slower
```

than FP32.

This is the weakest INT8 performance point among the tested workloads.

---

## 7. Matrix Size 512

At 512 × 512, the trend reverses.

Measured times:

```text
FP32:
693.5223 ms

INT8 kernel-only:
526.0128 ms

INT8 end-to-end:
533.7385 ms
```

Kernel speedup:

```text
1.3185×
```

End-to-end speedup:

```text
1.2994×
```

INT8 kernel-only latency reduction:

```text
1 - 526.0128 / 693.5223
≈ 24.15%
```

End-to-end latency reduction:

```text
1 - 533.7385 / 693.5223
≈ 23.04%
```

This is the strongest INT8 performance improvement in the experiment.

---

## 8. Matrix Size 1024

Measured times:

```text
FP32:
6794.1257 ms

INT8 kernel-only:
6250.8679 ms

INT8 end-to-end:
6313.0029 ms
```

Kernel speedup:

```text
1.0869×
```

End-to-end speedup:

```text
1.0762×
```

INT8 kernel-only latency reduction:

```text
≈ 8.00%
```

End-to-end latency reduction:

```text
≈ 7.08%
```

INT8 remains faster than FP32, but the advantage is smaller than at 512 × 512.

---

## 9. Performance Crossover

The experiment shows a clear workload-dependent crossover.

Small matrices:

```text
128:
INT8 slower

256:
INT8 slower
```

Larger matrices:

```text
512:
INT8 faster

1024:
INT8 faster
```

Therefore the crossover occurs between:

```text
256 × 256
```

and:

```text
512 × 512
```

for the current implementation.

---

## 10. Why INT8 Is Slower for Small Matrices

The current INT8 inner loop performs additional work such as:

```cpp
const int32_t a =
    static_cast<int32_t>(A[...])
    - zero_point;

const int32_t b =
    static_cast<int32_t>(B[...])
    - zero_point;

accumulator += a * b;
```

Compared with FP32, the current INT8 path includes:

```text
INT8 → INT32 conversion
zero-point subtraction
INT32 multiplication
INT32 accumulation
final FP32 scaling
```

The implementation does not yet use:

```text
SIMD INT8 dot-product instructions
VNNI
NEON dot-product
cache blocking
packed layouts
vectorized accumulation
```

For small matrices, this extra instruction overhead dominates, so INT8 is slower despite its smaller data representation.

---

## 11. Why INT8 Starts Winning at 512

As the workload grows, memory traffic and cache behavior become more important.

INT8 stores four times as many elements per byte as FP32:

```text
FP32:
4 bytes / value

INT8:
1 byte / value
```

At 512 × 512:

```text
FP32 input storage:
2.0 MB

INT8 input storage:
0.5 MB
```

The smaller working set improves cache utilization and reduces memory traffic enough for INT8 to become faster.

Measured kernel speedup:

```text
1.3185×
```

This is substantial even without architecture-specific INT8 optimization.

---

## 12. Why the 1024 Speedup Is Smaller Than 512

The strongest INT8 gain occurs at 512, not 1024.

This shows that performance does not improve monotonically with matrix size.

Possible contributing factors include:

```text
cache hierarchy transitions
memory access behavior
integer execution throughput
compiler vectorization differences
larger working-set pressure
conversion overhead
```

At 1024, both FP32 and INT8 workloads become much larger.

INT8 still benefits from reduced storage:

```text
FP32:
8 MB

INT8:
2 MB
```

but the naive integer implementation is not optimized enough to preserve the 512-level speedup.

---

## 13. Runtime Quantization Overhead

Quantization overhead can be approximated as:

```text
INT8 E2E - INT8 kernel
```

| Size | Kernel (ms) | E2E (ms) | Quantization Overhead |
|---:|---:|---:|---:|
| 128 | 7.6955 | 7.9469 | 0.2514 ms |
| 256 | 65.4724 | 67.9806 | 2.5082 ms |
| 512 | 526.0128 | 533.7385 | 7.7257 ms |
| 1024 | 6250.8679 | 6313.0029 | 62.1350 ms |

Approximate fraction of INT8 end-to-end time:

```text
128:
3.16%

256:
3.69%

512:
1.45%

1024:
0.98%
```

Therefore runtime quantization becomes proportionally less important as MatMul size grows.

This is why kernel-only and end-to-end speedups become increasingly close for larger matrices.

---

## 14. Weight Pre-Quantization Implication

In real inference, model weights are usually constant.

Therefore weights can often be quantized offline:

```text
FP32 weights
→ quantize once
→ store INT8 weights
```

In that case, inference does not need to pay the cost of quantizing the weight matrix for every request.

If B represents model weights, only activation A may need runtime quantization.

Therefore the current end-to-end benchmark is conservative compared with a realistic deployment using pre-quantized weights.

---

## 15. Numerical Accuracy

Measured MAE:

| Matrix Size | MAE |
|---:|---:|
| 128 | 0.016874 |
| 256 | 0.023688 |
| 512 | 0.033505 |
| 1024 | 0.047411 |

The error increases as the inner dimension grows.

This is expected because every output element contains more quantized multiply-accumulate operations.

More accumulated quantized products lead to more accumulated approximation error.

The results show a smooth error increase rather than unstable numerical behavior.

---

## 16. Memory Reduction

The raw input storage reduction remains:

```text
75%
```

for every matrix size.

Example at 1024:

```text
FP32 A + B:
8 MB

INT8 A + B:
2 MB
```

This is a guaranteed storage advantage independent of kernel speed.

Benefits include:

```text
smaller model storage
lower memory traffic
better cache utilization
potentially greater batch capacity
```

---

## 17. Overall Performance Pattern

The measured behavior can be summarized as:

```text
Small workload:
INT8 instruction/conversion overhead dominates
→ FP32 faster

Medium workload:
INT8 memory/cache advantage dominates
→ INT8 significantly faster

Large workload:
INT8 remains faster
→ but naive integer implementation limits the gain
```

Measured kernel speedups:

```text
128  → 0.7687×
256  → 0.7234×
512  → 1.3185×
1024 → 1.0869×
```

---

## 18. Main Findings

```text
1. INT8 MatMul is numerically stable enough for the current baseline.

2. INT8 is not automatically faster than FP32.

3. At 128 and 256, the naive INT8 implementation is slower than FP32.

4. A clear performance crossover occurs between 256 and 512.

5. At 512, INT8 kernel-only reaches 1.3185× speedup.

6. At 512, the full end-to-end path reaches 1.2994× speedup.

7. At 1024, INT8 remains faster with 1.0869× kernel speedup.

8. Runtime quantization overhead becomes proportionally smaller as workload size grows.

9. Raw input storage is reduced by exactly 75%.

10. MAE increases smoothly with matrix size due to longer accumulation lengths.

11. The current INT8 kernel does not yet exploit SIMD or architecture-specific dot-product instructions.

12. The next optimization stage should focus on OpenMP, cache-aware loop structure, and SIMD/vectorized INT8 accumulation.
```

---

## 19. Recommended Next Step

The current experiment establishes the single-thread baseline.

The next benchmark should compare:

```text
FP32 1 thread
FP32 2 threads
FP32 4 threads
FP32 8 threads

INT8 1 thread
INT8 2 threads
INT8 4 threads
INT8 8 threads
```

This will show whether the INT8 path scales across CPU workers as effectively as the FP32 implementation.

After that, the most important optimization is:

```text
SIMD / vectorized INT8 dot-product execution
```

because the current scalar implementation does not yet exploit the CPU's full INT8 arithmetic throughput.

---

## 20. Final Conclusion

The INT8 MatMul path provides a strong memory advantage and becomes faster than FP32 once the workload is sufficiently large.

The strongest measured result is:

```text
Matrix:
512 × 512

FP32:
693.5223 ms

INT8 kernel:
526.0128 ms

INT8 end-to-end:
533.7385 ms

Kernel speedup:
1.3185×

End-to-end speedup:
1.2994×

MAE:
0.033505

Memory reduction:
75%
```

The experiment demonstrates that INT8 performance is strongly workload-dependent.

For small matrices, integer-conversion and execution overhead dominate.

For larger matrices, the reduced memory footprint and improved cache behavior become valuable enough for INT8 to outperform FP32.

This provides the baseline for the next stage: multi-threaded and SIMD-accelerated INT8 MatMul.
