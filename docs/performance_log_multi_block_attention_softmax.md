# Attention Softmax Performance Analysis

## 1. Benchmark Setup

Two CUDA implementations of attention softmax were compared:

### Single-block version
- One CUDA block processes one softmax row.
- One kernel performs:
  - row max reduction
  - exponentiation
  - row sum reduction
  - normalization
- Block size: 256 threads.

### Multi-block version
- Multiple CUDA blocks process one softmax row.
- `blocks_per_row = 4`
- The computation is split into five kernels:
  1. `attention_softmax_partial_max_kernel`
  2. `attention_softmax_row_max_kernel`
  3. `attention_softmax_exp_sum_kernel`
  4. `attention_softmax_row_sum_kernel`
  5. `attention_softmax_normalize_kernel`

### Benchmark configuration
- Warmup runs: 20
- Benchmark runs: 100
- Timing method: CUDA Events
- Memory allocation and H2D copies are excluded from benchmark latency.
- GPU architecture target: `sm_70`

---

## 2. CUDA Event Benchmark Results

| seq_len | Single-block (ms) | Multi-block (ms) | Speedup (Single / Multi) |
|---:|---:|---:|---:|
| 128 | 0.004270 | 0.018401 | 0.232 |
| 256 | 0.004884 | 0.023091 | 0.212 |
| 512 | 0.007454 | 0.033321 | 0.224 |
| 1024 | 0.018063 | 0.058368 | 0.309 |
| 2048 | 0.071680 | 0.141220 | 0.508 |
| 4096 | 0.394895 | 0.453796 | 0.870 |
| 8192 | 1.537120 | 1.511220 | 1.017 |
| 12000 | 3.294710 | 3.171040 | 1.039 |

### Initial observation

The single-block implementation is significantly faster for small sequence lengths.

As `seq_len` increases, the performance gap gradually decreases.

The two implementations become approximately equal around `seq_len = 8192`.

The multi-block implementation becomes slightly faster for large sequence lengths:

- `seq_len = 8192`: approximately 1.7% faster.
- `seq_len = 12000`: approximately 3.9% faster.

Therefore, the initial crossover region is between:

```text
4096 < crossover <= 8192
```


# Nsight Systems Profiling Results

## 1. Single-Block Softmax — `seq_len = 4096`

### CUDA API Summary

| CUDA API | Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Median (ns) | Min (ns) | Max (ns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cudaMalloc` | 83.4 | 128,886,475 | 7 | 18,412,353.6 | 99,322 | 2,323 | 128,560,248 |
| `cudaMemcpy` | 10.3 | 15,930,554 | 1 | 15,930,554.0 | 15,930,554 | 15,930,554 | 15,930,554 |
| `cudaDeviceSynchronize` | 3.3 | 5,096,797 | 2 | 2,548,398.5 | 2,548,398.5 | 466,800 | 4,629,997 |
| `cudaFree` | 2.3 | 3,499,473 | 7 | 499,924.7 | 290,317 | 3,117 | 1,126,570 |
| `cudaLaunchKernel` | 0.8 | 1,207,840 | 11 | 109,803.6 | 8,070 | 5,609 | 1,095,920 |
| `cuModuleGetLoadingMode` | 0.0 | 1,535 | 1 | 1,535.0 | 1,535 | 1,535 | 1,535 |

### CUDA GPU Kernel Summary

| Kernel | Time (%) | Total Time (ns) | Instances | Avg (ns) | Median (ns) | Min (ns) | Max (ns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention_softmax_kernel` | 100.0 | 5,171,711 | 11 | 470,155.5 | 470,909 | 466,173 | 476,797 |

### CUDA GPU Memory Operations

| Operation | Total Time (ns) | Count | Avg (ns) | Total Size (MB) |
|---|---:|---:|---:|---:|
| Host-to-Device memcpy | 14,823,107 | 1 | 14,823,107.0 | 67.109 |

---

## 2. Multi-Block Softmax — `seq_len = 4096`

### CUDA API Summary

| CUDA API | Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Median (ns) | Min (ns) | Max (ns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cudaMalloc` | 78.1 | 84,444,396 | 7 | 12,063,485.1 | 71,172 | 1,523 | 84,160,469 |
| `cudaMemcpy` | 12.1 | 13,128,216 | 1 | 13,128,216.0 | 13,128,216 | 13,128,216 | 13,128,216 |
| `cudaDeviceSynchronize` | 5.1 | 5,533,500 | 2 | 2,766,750.0 | 2,766,750 | 508,123 | 5,025,377 |
| `cudaFree` | 4.1 | 4,440,948 | 7 | 634,421.1 | 170,314 | 1,849 | 2,141,864 |
| `cudaLaunchKernel` | 0.5 | 567,804 | 55 | 10,323.7 | 4,431 | 3,435 | 263,741 |
| `cuModuleGetLoadingMode` | 0.0 | 1,874 | 1 | 1,874.0 | 1,874 | 1,874 | 1,874 |

### CUDA GPU Kernel Summary

| Kernel | Time (%) | Total Time (ns) | Instances | Avg (ns) | Median (ns) | Min (ns) | Max (ns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention_softmax_exp_sum_kernel` | 36.2 | 2,093,128 | 11 | 190,284.4 | 190,205 | 188,862 | 191,966 |
| `attention_softmax_normalize_kernel` | 34.5 | 1,992,138 | 11 | 181,103.5 | 181,118 | 180,126 | 182,014 |
| `attention_softmax_partial_max_kernel` | 23.5 | 1,360,273 | 11 | 123,661.2 | 123,295 | 122,143 | 128,926 |
| `attention_softmax_row_sum_kernel` | 2.9 | 167,326 | 11 | 15,211.5 | 15,135 | 15,008 | 15,647 |
| `attention_softmax_row_max_kernel` | 2.9 | 165,599 | 11 | 15,054.5 | 15,040 | 14,784 | 15,456 |

### Estimated Per-Execution GPU Time

| Component | Avg Time (us) |
|---|---:|
| Partial max | 123.661 |
| Row max | 15.055 |
| Exp + partial sum | 190.284 |
| Row sum | 15.212 |
| Normalize | 181.104 |
| **Total** | **525.315** |

### CUDA GPU Memory Operations

| Operation | Total Time (ns) | Count | Avg (ns) | Total Size (MB) |
|---|---:|---:|---:|---:|
| Host-to-Device memcpy | 12,971,121 | 1 | 12,971,121.0 | 67.109 |

---

## 3. Single-Block Softmax — `seq_len = 8192`

### CUDA API Summary

| CUDA API | Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Median (ns) | Min (ns) | Max (ns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cudaMalloc` | 42.6 | 66,809,795 | 7 | 9,544,256.4 | 71,845 | 1,913 | 66,547,218 |
| `cudaMemcpy` | 40.7 | 63,869,029 | 1 | 63,869,029.0 | 63,869,029 | 63,869,029 | 63,869,029 |
| `cudaDeviceSynchronize` | 12.9 | 20,323,261 | 2 | 10,161,630.5 | 10,161,630.5 | 1,857,474 | 18,465,787 |
| `cudaFree` | 3.6 | 5,679,273 | 7 | 811,324.7 | 362,043 | 2,604 | 2,272,941 |
| `cudaLaunchKernel` | 0.2 | 283,918 | 11 | 25,810.7 | 4,132 | 3,330 | 220,134 |
| `cuModuleGetLoadingMode` | 0.0 | 1,090 | 1 | 1,090.0 | 1,090 | 1,090 | 1,090 |

### CUDA GPU Kernel Summary

| Kernel | Time (%) | Total Time (ns) | Instances | Avg (ns) | Median (ns) | Min (ns) | Max (ns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention_softmax_kernel` | 100.0 | 20,351,695 | 11 | 1,850,154.1 | 1,849,127 | 1,845,223 | 1,856,551 |

### CUDA GPU Memory Operations

| Operation | Total Time (ns) | Count | Avg (ns) | Total Size (MB) |
|---|---:|---:|---:|---:|
| Host-to-Device memcpy | 63,647,980 | 1 | 63,647,980.0 | 268.435 |

---

## 4. Multi-Block Softmax — `seq_len = 8192`

### CUDA API Summary

| CUDA API | Time (%) | Total Time (ns) | Num Calls | Avg (ns) | Median (ns) | Min (ns) | Max (ns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cudaMalloc` | 53.0 | 88,696,576 | 7 | 12,670,939.4 | 85,258 | 2,125 | 88,368,840 |
| `cudaMemcpy` | 31.8 | 53,316,776 | 1 | 53,316,776.0 | 53,316,776 | 53,316,776 | 53,316,776 |
| `cudaDeviceSynchronize` | 12.0 | 20,051,337 | 2 | 10,025,668.5 | 10,025,668.5 | 1,829,327 | 18,222,010 |
| `cudaFree` | 2.8 | 4,676,102 | 7 | 668,014.6 | 419,327 | 2,707 | 2,130,845 |
| `cudaLaunchKernel` | 0.4 | 716,630 | 55 | 13,029.6 | 4,712 | 3,561 | 370,949 |
| `cuModuleGetLoadingMode` | 0.0 | 2,380 | 1 | 2,380.0 | 2,380 | 2,380 | 2,380 |

### CUDA GPU Kernel Summary

| Kernel | Time (%) | Total Time (ns) | Instances | Avg (ns) | Median (ns) | Min (ns) | Max (ns) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `attention_softmax_exp_sum_kernel` | 38.5 | 7,819,039 | 11 | 710,821.7 | 710,927 | 709,166 | 712,462 |
| `attention_softmax_normalize_kernel` | 37.5 | 7,628,827 | 11 | 693,529.7 | 693,550 | 692,846 | 694,958 |
| `attention_softmax_partial_max_kernel` | 21.0 | 4,270,039 | 11 | 388,185.4 | 387,688 | 387,304 | 392,968 |
| `attention_softmax_row_sum_kernel` | 1.5 | 303,142 | 11 | 27,558.4 | 27,521 | 27,360 | 27,873 |
| `attention_softmax_row_max_kernel` | 1.5 | 299,878 | 11 | 27,261.6 | 27,169 | 27,008 | 27,681 |

### Estimated Per-Execution GPU Time

| Component | Avg Time (us) |
|---|---:|
| Partial max | 388.185 |
| Row max | 27.262 |
| Exp + partial sum | 710.822 |
| Row sum | 27.558 |
| Normalize | 693.530 |
| **Total** | **1,847.357** |

### CUDA GPU Memory Operations

| Operation | Total Time (ns) | Count | Avg (ns) | Total Size (MB) |
|---|---:|---:|---:|---:|
| Host-to-Device memcpy | 53,067,477 | 1 | 53,067,477.0 | 268.435 |

---

## 5. Cross-Configuration Comparison

| Configuration | Avg GPU Time |
|---|---:|
| Single-block, `seq_len = 4096` | 470.156 us |
| Multi-block, `seq_len = 4096` | 525.315 us |
| Single-block, `seq_len = 8192` | 1,850.154 us |
| Multi-block, `seq_len = 8192` | 1,847.357 us |

### Relative Performance

| seq_len | Single (us) | Multi (us) | Multi vs Single |
|---:|---:|---:|---:|
| 4096 | 470.156 | 525.315 | 11.7% slower |
| 8192 | 1,850.154 | 1,847.357 | 0.15% faster |

### Scaling from 4096 to 8192

| Implementation | 4096 Time (us) | 8192 Time (us) | Scaling Factor |
|---|---:|---:|---:|
| Single-block | 470.156 | 1,850.154 | 3.94x |
| Multi-block | 525.315 | 1,847.357 | 3.52x |




# Nsight Systems Analysis

## 1. Main Observation

The Nsight Systems profiling results confirm the crossover behavior previously observed in the CUDA Event benchmark.

At `seq_len = 4096`, the single-block implementation is still clearly faster:

Single-block: 470.156 us  
Multi-block: 525.315 us

The multi-block implementation is approximately 11.7% slower.

At `seq_len = 8192`, the two implementations become almost identical in performance:

Single-block: 1850.154 us  
Multi-block: 1847.357 us

Under Nsight Systems profiling, the multi-block implementation is only about 0.15% faster.

This confirms that `seq_len = 8192` is inside the crossover region rather than being a point where the multi-block implementation suddenly becomes significantly faster.

---

## 2. Multi-Block Overhead at `seq_len = 4096`

The multi-block implementation splits the softmax computation into five kernels:

- `attention_softmax_partial_max_kernel`
- `attention_softmax_row_max_kernel`
- `attention_softmax_exp_sum_kernel`
- `attention_softmax_row_sum_kernel`
- `attention_softmax_normalize_kernel`

At `seq_len = 4096`, the two lightweight reduction kernels take:

- `row_max = 15.055 us`
- `row_sum = 15.212 us`

Their combined execution time is:

30.267 us

The total multi-block GPU execution time is approximately:

525.315 us

Therefore, the two auxiliary reduction kernels account for:

30.267 / 525.315 ≈ 5.76%

of the total multi-block execution time.

These kernels perform relatively little useful computation because each row only needs to reduce `blocks_per_row = 4` partial values.

As a result, a significant portion of their execution cost comes from structural overhead such as:

- additional kernel launches
- intermediate global-memory communication
- synchronization between kernel stages
- accesses to intermediate buffers such as `partial_max`, `partial_sum`, `row_max`, and `row_sum`

At `seq_len = 4096`, these additional costs are still large enough to make the multi-block implementation slower than the single-block implementation.

---

## 3. Overhead Amortization at `seq_len = 8192`

At `seq_len = 8192`, the same two auxiliary reduction kernels take:

- `row_max = 27.262 us`
- `row_sum = 27.558 us`

Their combined execution time becomes:

54.820 us

The total multi-block execution time is approximately:

1847.357 us

Therefore, their relative contribution becomes:

54.820 / 1847.357 ≈ 2.97%

The relative overhead therefore decreases from:

5.76% at `seq_len = 4096`

to:

2.97% at `seq_len = 8192`

Although the absolute execution time of these reduction kernels increases, their percentage of the total runtime decreases significantly.

This is an example of overhead amortization.

The synchronization and extra-kernel costs do not disappear, but as the workload becomes larger, these fixed or semi-fixed costs become less important relative to the amount of useful computation.

---

## 4. Scaling Behavior

When the sequence length increases from 4096 to 8192, the attention score matrix grows from:

4096 × 4096

to:

8192 × 8192

Because the matrix contains `seq_len^2` elements, doubling the sequence length increases the total number of elements by approximately 4x.

For the single-block implementation, the runtime increases from:

470.156 us

to:

1850.154 us

The scaling factor is:

1850.154 / 470.156 ≈ 3.94x

This is very close to the theoretical 4x increase in workload.

For the multi-block implementation, the runtime increases from:

525.315 us

to:

1847.357 us

The scaling factor is:

1847.357 / 525.315 ≈ 3.52x

Therefore:

- Single-block scaling: approximately 3.94x
- Multi-block scaling: approximately 3.52x

This indicates that the multi-block implementation scales more efficiently as the workload increases.

---

## 5. Heavy-Kernel Scaling

The three kernels responsible for most of the useful computation in the multi-block implementation are:

- `attention_softmax_partial_max_kernel`
- `attention_softmax_exp_sum_kernel`
- `attention_softmax_normalize_kernel`

At `seq_len = 4096`, their combined runtime is:

123.661 + 190.284 + 181.104 = 495.049 us

At `seq_len = 8192`, their combined runtime is:

388.185 + 710.822 + 693.530 = 1792.537 us

The scaling factor is:

1792.537 / 495.049 ≈ 3.62x

This is lower than the single-block scaling factor of approximately 3.94x.

This suggests that the kernels performing the actual matrix traversal benefit from the additional per-row parallelism introduced by the multi-block design.

---

## 6. Per-Thread Workload Difference

The single-block implementation assigns one CUDA block to each softmax row.

The block size is 256 threads.

At `seq_len = 4096`:

4096 / 256 ≈ 16 elements per thread

At `seq_len = 8192`:

8192 / 256 ≈ 32 elements per thread

Therefore, when the sequence length doubles, each thread in the single-block implementation performs approximately twice as much serial work inside its loop.

This means that the amount of intra-row parallelism is limited by the single CUDA block assigned to each row.

In the multi-block implementation:

`blocks_per_row = 4`

Therefore, each row is distributed across four CUDA blocks.

The approximate effective workload per thread becomes:

At `seq_len = 4096`:

4096 / (256 × 4) ≈ 4 elements per thread

At `seq_len = 8192`:

8192 / (256 × 4) ≈ 8 elements per thread

This significantly reduces per-thread serial work and exposes more parallelism within each row.

The advantage becomes increasingly important as the row length increases.

---

## 7. Why Multi-Block Is Slower for Small Inputs

For small sequence lengths, the single-block implementation has several structural advantages:

- only one kernel is required
- reductions can be completed within one block
- shared memory can be reused directly
- no intermediate global-memory buffers are required
- no cross-kernel synchronization is required
- fewer intermediate reads and writes are performed

The multi-block implementation introduces additional overhead in exchange for higher intra-row parallelism.

This includes:

- five kernel stages instead of one
- extra kernel launch overhead
- intermediate buffers
- additional global-memory traffic
- synchronization boundaries between kernel stages

For small rows, the additional parallelism is unnecessary because one CUDA block already has enough threads to process the row efficiently.

Therefore, the overhead dominates and the multi-block implementation is slower.

---

## 8. Why the Crossover Appears

The measured behavior can be summarized as follows:

Small `seq_len`  
→ Multi-block overhead dominates  
→ Single-block is significantly faster

As `seq_len` increases  
→ The amount of serial work per thread in the single-block implementation increases  
→ The additional parallelism of the multi-block implementation becomes more useful  
→ The relative cost of extra reduction and synchronization stages decreases  
→ The performance gap becomes smaller

At approximately `seq_len = 8192`  
→ The two implementations reach similar performance  
→ A crossover region appears

The crossover is therefore caused by a trade-off between two design strategies.

Single-block:

- lower overhead
- fewer kernel launches
- less synchronization
- limited per-row parallelism

Multi-block:

- higher overhead
- more kernel launches
- more intermediate communication
- higher per-row parallelism

At small sequence lengths, reducing overhead is more important.

At large sequence lengths, exposing more parallelism becomes increasingly valuable.

---

## 9. CUDA API Results

The CUDA API summary shows that the multi-block implementation launches significantly more kernels.

For the single-block profiling run:

11 `cudaLaunchKernel` calls were recorded.

This corresponds to:

10 warmup runs + 1 profiling run

For the multi-block profiling run:

55 `cudaLaunchKernel` calls were recorded.

Because each multi-block softmax execution consists of five kernels:

11 executions × 5 kernels = 55 kernel launches

This confirms that the profiling results match the expected program structure.

The CUDA API summary also contains large values for:

- `cudaMalloc`
- `cudaMemcpy`
- `cudaFree`

These values should not be used when comparing softmax kernel performance.

They belong to initialization and cleanup stages and are excluded from the CUDA Event benchmark.

The primary performance comparison should therefore focus on GPU kernel execution time.

---

## 10. Relationship Between CUDA Event Benchmark and Nsight Systems

The CUDA Event benchmark and Nsight Systems produce slightly different absolute execution times.

For example, at `seq_len = 8192`, the CUDA Event benchmark measured approximately:

Single-block: 1.537 ms  
Multi-block: 1.511 ms

The Nsight Systems profiling results measured approximately:

Single-block: 1.850 ms  
Multi-block: 1.847 ms

The difference is expected because profiling introduces additional tracing overhead.

Therefore:

- CUDA Event timing should be treated as the primary performance measurement.
- Nsight Systems should be used as profiling evidence to understand where execution time is spent.

The relative trend is more important than the exact profiled latency.

Both methods show that:

- single-block is faster at `seq_len = 4096`
- the two implementations converge near `seq_len = 8192`
- multi-block begins to become competitive at large sequence lengths

---

## 11. Current Conclusion

The Nsight Systems results confirm the crossover trend observed in the original CUDA Event benchmark.

The main findings are:

1. At `seq_len = 4096`, the multi-block implementation is approximately 11.7% slower than the single-block implementation.

2. At `seq_len = 8192`, the two implementations are effectively tied under Nsight Systems profiling.

3. The relative contribution of the two auxiliary reduction kernels decreases from approximately 5.76% to 2.97%.

4. The single-block implementation scales by approximately 3.94x when `seq_len` doubles from 4096 to 8192.

5. The multi-block implementation scales by approximately 3.52x over the same range.

6. The three main multi-block computation kernels scale by approximately 3.62x.

7. The additional synchronization and kernel-launch overhead becomes relatively less important as the workload increases.

8. The multi-block implementation provides higher per-row parallelism and reduces the amount of serial work performed by each thread.

These results support the hypothesis that multi-block execution provides better scalability for long softmax rows.

However, Nsight Systems only shows where GPU execution time is spent.

It does not directly reveal the microarchitectural reason for the improved scaling.

---

## 12. Next Step: Nsight Compute

The next step is to use NVIDIA Nsight Compute to compare the hardware behavior of the single-block and multi-block implementations.

The initial comparison should focus on:

- Single-block, `seq_len = 4096`
- Single-block, `seq_len = 8192`
- Multi-block, `seq_len = 4096`
- Multi-block, `seq_len = 8192`

Important metrics include:

- SM Throughput
- Memory Throughput
- Achieved Occupancy
- DRAM Throughput
- L2 Throughput
- Eligible Warps Per Cycle
- Warp Stall Reasons

The goal is to explain why:

Single-block scaling ≈ 3.94x

while:

Multi-block scaling ≈ 3.52x

when the sequence length doubles.

The key question is whether the improved multi-block scaling comes from:

- higher SM utilization
- better occupancy
- increased memory-level parallelism
- higher DRAM or L2 throughput
- reduced per-thread serial work
- improved warp scheduling
- or a combination of these factors