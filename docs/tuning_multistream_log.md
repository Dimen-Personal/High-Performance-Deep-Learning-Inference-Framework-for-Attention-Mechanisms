# Nsight Systems SQLite Analysis — Multi-Stream Attention

## 1. SQLite Data Used

The analysis is based only on the following exported Nsight Systems SQLite files:

```text
attention_multistream_256.sqlite
attention_multistream_2048.sqlite
```

The profiling program performs:

```text
2 warmup runs
+
3 profiling runs
=
5 total captured runs
```

The SQLite files therefore contain all 5 executions.  
For the analysis below, only the **final 3 runs** are treated as the profiling region.

Two CUDA streams are present:

```text
Stream 13
Stream 14
```

Each stream executes one complete Attention request:

```text
H2D Q
H2D K
H2D V
↓
AttentionScores
↓
AttentionSoftmax
↓
MatMul
↓
D2H Output
```

---

# 2. SQLite Results — SeqLen = 256

## 2.1 Kernel Timing

Average kernel duration over the final 3 profiling runs:

| Stream | Kernel | Average Duration |
|---:|---|---:|
| 13 | AttentionScores | 28.830 µs |
| 13 | AttentionSoftmax | 4.639 µs |
| 13 | MatMul | 21.801 µs |
| 14 | AttentionScores | 28.797 µs |
| 14 | AttentionSoftmax | 4.469 µs |
| 14 | MatMul | 12.842 µs |

### Stream 13 MatMul individual durations

```text
Run 1: 12.831 µs
Run 2: 33.949 µs
Run 3: 18.623 µs
```

Average:

```text
21.801 µs
```

### Stream 14 MatMul individual durations

```text
Run 1: 12.927 µs
Run 2: 12.735 µs
Run 3: 12.863 µs
```

Average:

```text
12.842 µs
```

---

## 2.2 Stream Overlap

Measured overlap during the final 3 profiling runs:

| Overlap Type | Duration |
|---|---:|
| Total Stream 13 / Stream 14 overlap | 0.110804 ms |
| Kernel ↔ Kernel overlap | 0.037596 ms |
| Stream 13 Copy ↔ Stream 14 Kernel | 0.011070 ms |
| Stream 14 Copy ↔ Stream 13 Kernel | 0.058458 ms |

Therefore:

```text
Total stream overlap:
0.110804 ms
```

of which:

```text
Kernel-kernel overlap:
0.037596 ms
```

and significant copy-compute overlap is also present.

---

# 3. SQLite Results — SeqLen = 2048

## 3.1 Kernel Timing

Average kernel duration over the final 3 profiling runs:

| Stream | Kernel | Average Duration |
|---:|---|---:|
| 13 | AttentionScores | 793.396 µs |
| 13 | AttentionSoftmax | 89.486 µs |
| 13 | MatMul | 459.664 µs |
| 14 | AttentionScores | 830.449 µs |
| 14 | AttentionSoftmax | 108.439 µs |
| 14 | MatMul | 641.963 µs |

### Stream 13 MatMul individual durations

```text
Run 1: 494.711 µs
Run 2: 442.172 µs
Run 3: 442.108 µs
```

Average:

```text
459.664 µs
```

### Stream 14 MatMul individual durations

```text
Run 1: 685.928 µs
Run 2: 589.071 µs
Run 3: 650.890 µs
```

Average:

```text
641.963 µs
```

---

## 3.2 Stream Overlap

Measured overlap during the final 3 profiling runs:

| Overlap Type | Duration |
|---|---:|
| Total Stream 13 / Stream 14 overlap | 2.223689 ms |
| Kernel ↔ Kernel overlap | 1.696692 ms |
| Stream 13 Copy ↔ Stream 14 Kernel | 0.124214 ms |
| Stream 14 Copy ↔ Stream 13 Kernel | 0.402783 ms |

Therefore:

```text
Total stream overlap:
2.223689 ms
```

of which:

```text
Kernel-kernel overlap:
1.696692 ms
```

---

# 4. Direct 256 vs 2048 Comparison

## 4.1 Overlap Comparison

| Metric | SeqLen 256 | SeqLen 2048 |
|---|---:|---:|
| Total stream overlap | 0.110804 ms | 2.223689 ms |
| Kernel-kernel overlap | 0.037596 ms | 1.696692 ms |
| Stream 13 Copy ↔ Stream 14 Kernel | 0.011070 ms | 0.124214 ms |
| Stream 14 Copy ↔ Stream 13 Kernel | 0.058458 ms | 0.402783 ms |

The most important observation is:

```text
SeqLen 2048 has much more absolute overlap than SeqLen 256.
```

Kernel-kernel overlap increases from:

```text
0.037596 ms
```

to:

```text
1.696692 ms
```

This is approximately:

```text
45.1× more absolute kernel overlap
```

Therefore, the lower multi-stream scaling at large sequence length cannot be explained by a lack of concurrency.

---

# 5. Analysis Process

## Step 1 — Verify that two CUDA streams are actually active

The SQLite data contains GPU activities on:

```text
Stream 13
Stream 14
```

Both streams independently execute:

```text
AttentionScores
AttentionSoftmax
MatMul
```

Their timestamp intervals overlap.

### Conclusion

The implementation is achieving real GPU-side request-level concurrency.

The multi-stream behavior is not merely CPU-side asynchronous submission.

---

## Step 2 — Check whether kernels overlap

For SeqLen 256:

```text
Kernel-kernel overlap:
0.037596 ms
```

For SeqLen 2048:

```text
Kernel-kernel overlap:
1.696692 ms
```

### Conclusion

CUDA kernels from the two independent Attention requests execute concurrently.

Furthermore, larger Attention workloads create substantially more absolute kernel overlap.

---

## Step 3 — Check whether memory copies overlap computation

For SeqLen 256:

```text
Stream 13 Copy ↔ Stream 14 Kernel:
0.011070 ms

Stream 14 Copy ↔ Stream 13 Kernel:
0.058458 ms
```

For SeqLen 2048:

```text
Stream 13 Copy ↔ Stream 14 Kernel:
0.124214 ms

Stream 14 Copy ↔ Stream 13 Kernel:
0.402783 ms
```

### Conclusion

The pinned-memory + `cudaMemcpyAsync` implementation is producing real copy-compute overlap.

Therefore, part of the benefit of the multi-stream implementation comes from overlapping:

```text
memory transfer
+
GPU computation
```

---

## Step 4 — Test the original hypothesis

The original simple hypothesis was:

```text
Larger seq_len
→ less overlap
→ lower multi-stream speedup
```

The SQLite data contradicts this.

The 2048 workload has:

```text
much more total overlap
```

and:

```text
much more kernel-kernel overlap
```

than the 256 workload.

Therefore:

```text
lower large-sequence scaling is NOT caused by reduced overlap
```

---

## Step 5 — Inspect kernel latency under concurrency

The strongest difference appears in MatMul.

For SeqLen 2048:

```text
Stream 13 MatMul average:
459.664 µs
```

while:

```text
Stream 14 MatMul average:
641.963 µs
```

Difference:

```text
182.299 µs
```

Relative difference:

```text
~39.7% longer on Stream 14
```

AttentionSoftmax also differs:

```text
Stream 13:
89.486 µs

Stream 14:
108.439 µs
```

Relative difference:

```text
~21.2% longer on Stream 14
```

AttentionScores is more stable:

```text
Stream 13:
793.396 µs

Stream 14:
830.449 µs
```

Relative difference:

```text
~4.7%
```

---

## Step 6 — Interpret the kernel timing difference

The SQLite timestamps show that the large kernels execute concurrently for substantial periods.

This means that the GPU must service work from both streams at the same time.

Concurrency does not create additional GPU hardware.

The two requests may need to share resources such as:

```text
SM execution capacity
warp scheduling capacity
register resources
cache resources
memory bandwidth
memory subsystem resources
```

The current Nsight Systems SQLite data cannot determine exactly which hardware resource is saturated.

However, it does show the following pattern:

```text
larger workloads
→ substantially more concurrent execution
→ individual kernel durations become less symmetric
→ MatMul shows the strongest increase
```

This pattern is consistent with increasing GPU resource contention.

---

# 6. Final Summary

## Confirmed from SQLite

The SQLite data confirms all of the following:

```text
1. Two independent CUDA streams are active.

2. Attention kernels from the two streams overlap in time.

3. Memory copies and GPU kernels overlap in time.

4. SeqLen 2048 has significantly more absolute overlap than SeqLen 256.

5. Therefore, the reduced scaling at SeqLen 2048 is not caused by insufficient concurrency.

6. MatMul shows the strongest timing degradation/asymmetry during large concurrent workloads.

7. AttentionSoftmax also shows noticeable degradation.

8. AttentionScores is comparatively more stable.
```

---

## Key Numbers

### SeqLen 256

```text
Total stream overlap:
0.110804 ms

Kernel-kernel overlap:
0.037596 ms

Stream 13 Copy ↔ Stream 14 Kernel:
0.011070 ms

Stream 14 Copy ↔ Stream 13 Kernel:
0.058458 ms
```

### SeqLen 2048

```text
Total stream overlap:
2.223689 ms

Kernel-kernel overlap:
1.696692 ms

Stream 13 Copy ↔ Stream 14 Kernel:
0.124214 ms

Stream 14 Copy ↔ Stream 13 Kernel:
0.402783 ms
```

### SeqLen 2048 Kernel Averages

```text
AttentionScores:
Stream 13 = 793.396 µs
Stream 14 = 830.449 µs

AttentionSoftmax:
Stream 13 = 89.486 µs
Stream 14 = 108.439 µs

MatMul:
Stream 13 = 459.664 µs
Stream 14 = 641.963 µs
```

---

## Final Interpretation

The data supports the following model:

```text
Small / medium Attention workload
        ↓
enough work exists for useful overlap
        ↓
GPU still has available execution headroom
        ↓
multi-stream concurrency improves throughput efficiently
```

but for a larger Attention workload:

```text
large kernels
        ↓
very strong stream overlap
        ↓
both requests compete for finite GPU resources
        ↓
individual kernel execution becomes slower
        ↓
additional overlap produces diminishing throughput benefit
```

Therefore:

```text
More overlap does not automatically mean more speedup.
```

The central conclusion from the current SQLite analysis is:

> The multi-stream implementation is successfully creating real GPU concurrency. At large sequence lengths, the limiting factor is no longer the ability to overlap work; the evidence instead points toward increasing resource contention during concurrent execution, with MatMul showing the strongest effect.


#
#
#
#
#

# MatMul Block Configuration Tuning Report

## 1. Objective

The purpose of this experiment is to determine whether the original CUDA MatMul block configuration can be improved through empirical tuning.

The current implementation uses:

```cpp
dim3 block(16, 16);
```

Because Nsight Compute hardware performance counters are unavailable in the current environment, the tuning process is performed experimentally. The mathematical implementation of the kernel remains unchanged; only the CUDA block shape is modified.

Tested configurations:

```text
16 × 16
8 × 8
32 × 8
8 × 32
```

The original `16 × 16` configuration is used as the baseline.

---

## 2. MatMul Workload

The MatMul corresponds to the final matrix multiplication in the Attention pipeline:

```text
weights: [seq_len, seq_len]
V:       [seq_len, 64]
output:  [seq_len, 64]
```

Therefore:

```text
M = seq_len
K = seq_len
N = 64
```

Kernel computation:

```cpp
for(int k = 0; k < K; k++)
{
    sum += A[row * K + k] * B[k * N + col];
}
```

---

## 3. Experimental Setup

Sequence lengths:

```text
128
256
512
1024
2048
```

Fixed settings:

```text
Head Dimension: 64
Warmup Runs: 10
Benchmark Runs: 100
```

Tested block configurations:

| Block | Threads/Block |
|---|---:|
| 16×16 | 256 |
| 8×8 | 64 |
| 32×8 | 256 |
| 8×32 | 256 |

All configurations passed correctness validation.

`vs16x16` is calculated as:

```text
baseline_time / tested_time
```

Therefore:

```text
> 1.0 = faster than 16×16
= 1.0 = same as 16×16
< 1.0 = slower than 16×16
```

---

## 4. Experimental Results

### SeqLen = 128

| Block | Threads | Time (ms) | Relative to 16×16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.007619 | 1.0000× | PASS |
| 8×8 | 64 | 0.007690 | 0.9907× | PASS |
| 32×8 | 256 | 0.007649 | 0.9960× | PASS |
| 8×32 | 256 | 0.008418 | 0.9051× | PASS |

Best configuration:

```text
16 × 16
```

---

### SeqLen = 256

| Block | Threads | Time (ms) | Relative to 16×16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.013312 | 1.0000× | PASS |
| 8×8 | 64 | 0.014520 | 0.9168× | PASS |
| 32×8 | 256 | 0.013557 | 0.9819× | PASS |
| 8×32 | 256 | 0.014694 | 0.9059× | PASS |

Best configuration:

```text
16 × 16
```

---

### SeqLen = 512

| Block | Threads | Time (ms) | Relative to 16×16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.028867 | 1.0000× | PASS |
| 8×8 | 64 | 0.036854 | 0.7833× | PASS |
| 32×8 | 256 | 0.026081 | 1.1068× | PASS |
| 8×32 | 256 | 0.039987 | 0.7219× | PASS |

Best configuration:

```text
32 × 8
```

Speedup over 16×16:

```text
1.1068×
```

Time reduction:

```text
1 - 0.026081 / 0.028867
≈ 9.65%
```

---

### SeqLen = 1024

| Block | Threads | Time (ms) | Relative to 16×16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.103813 | 1.0000× | PASS |
| 8×8 | 64 | 0.122522 | 0.8473× | PASS |
| 32×8 | 256 | 0.093778 | 1.1070× | PASS |
| 8×32 | 256 | 0.142459 | 0.7287× | PASS |

Best configuration:

```text
32 × 8
```

Speedup:

```text
1.1070×
```

Time reduction:

```text
1 - 0.093778 / 0.103813
≈ 9.67%
```

---

### SeqLen = 2048

| Block | Threads | Time (ms) | Relative to 16×16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.295557 | 1.0000× | PASS |
| 8×8 | 64 | 0.394424 | 0.7493× | PASS |
| 32×8 | 256 | 0.268605 | 1.1003× | PASS |
| 8×32 | 256 | 0.418867 | 0.7056× | PASS |

Best configuration:

```text
32 × 8
```

Speedup:

```text
1.1003×
```

Time reduction:

```text
1 - 0.268605 / 0.295557
≈ 9.12%
```

---

## 5. Best Configuration Summary

| SeqLen | Best Block | Time (ms) | Speedup vs 16×16 |
|---:|---|---:|---:|
| 128 | 16×16 | 0.007619 | 1.0000× |
| 256 | 16×16 | 0.013312 | 1.0000× |
| 512 | 32×8 | 0.026081 | 1.1068× |
| 1024 | 32×8 | 0.093778 | 1.1070× |
| 2048 | 32×8 | 0.268605 | 1.1003× |

Observed transition:

```text
SeqLen <= 256
→ 16×16 performs best

SeqLen >= 512
→ 32×8 performs best
```

---

## 6. Analysis

### 6.1 Why Block Shape Matters

CUDA executes threads in groups of 32 threads, called a warp.

Although several tested configurations use the same total number of threads per block:

```text
16×16 = 256
32×8  = 256
8×32  = 256
```

their performance differs significantly.

Therefore:

```text
threads per block alone does not determine performance
```

The layout of threads in x and y changes how a warp maps onto rows and columns, which changes memory access behavior.

---

### 6.2 Why 32×8 Performs Better for Large Workloads

For:

```text
block = dim3(32, 8)
```

the x dimension contains exactly 32 threads, equal to one warp.

One warp therefore approximately maps to:

```text
same output row
+
32 consecutive output columns
```

Conceptually:

```text
Thread 0  → row 0, col 0
Thread 1  → row 0, col 1
...
Thread 31 → row 0, col 31
```

During:

```cpp
sum += A[row * K + k] * B[k * N + col];
```

all threads in the warp use approximately the same `A[row][k]` while reading consecutive values from B:

```text
B[k][0]
B[k][1]
...
B[k][31]
```

This creates a favorable coalesced memory access pattern.

At larger workloads, this mapping becomes increasingly beneficial.

---

### 6.3 Why 16×16 Is Slightly Better for Small Workloads

For:

```text
block = dim3(16, 16)
```

one warp spans approximately two rows:

```text
Threads 0–15  → row 0
Threads 16–31 → row 1
```

This is still a reasonable mapping, and for small workloads the kernel execution time is extremely short.

At SeqLen 128:

```text
16×16 = 0.007619 ms
32×8  = 0.007649 ms
```

Difference:

```text
~0.4%
```

At SeqLen 256:

```text
16×16 = 0.013312 ms
32×8  = 0.013557 ms
```

Difference:

```text
~1.8%
```

At these scales, launch and scheduling overheads make up a larger fraction of total execution time, so the memory-layout advantage of 32×8 does not yet dominate.

---

### 6.4 Why 8×32 Performs Poorly

For:

```text
block = dim3(8, 32)
```

one warp spans approximately four rows:

```text
Threads 0–7   → row 0
Threads 8–15  → row 1
Threads 16–23 → row 2
Threads 24–31 → row 3
```

This is less favorable for the current MatMul memory access pattern.

At SeqLen 2048:

```text
32×8 = 0.268605 ms
8×32 = 0.418867 ms
```

So 8×32 is approximately:

```text
0.418867 / 0.268605 ≈ 1.56×
```

as slow, or about 56% slower.

This is important because both use exactly 256 threads per block.

---

### 6.5 Why 8×8 Performs Poorly

The 8×8 configuration uses only:

```text
64 threads/block
```

or two warps per block.

Its x dimension is also only 8 threads, so each warp spans multiple rows.

At SeqLen 2048:

```text
8×8  = 0.394424 ms
32×8 = 0.268605 ms
```

Therefore 8×8 is approximately 47% slower than the best configuration.

---

## 7. Performance Trend

The experiment shows:

```text
Small workload:
16×16 ≈ 32×8

Medium workload:
32×8 begins to win

Large workload:
32×8 remains consistently ~10% faster
```

The measured crossover occurs between:

```text
SeqLen = 256
```

and:

```text
SeqLen = 512
```

The consistency of the 32×8 improvement at 512, 1024, and 2048 suggests that this is a structural performance effect rather than measurement noise.

---

## 8. Recommended Runtime Policy

Recommended launch policy:

```cpp
dim3 block;

if(M <= 256)
{
    block = dim3(16, 16);
}
else
{
    block = dim3(32, 8);
}
```

For the current Attention MatMul:

```text
M = seq_len
```

so this becomes:

```cpp
dim3 block;

if(seq_len <= 256)
{
    block = dim3(16, 16);
}
else
{
    block = dim3(32, 8);
}
```

The grid calculation remains:

```cpp
dim3 grid(
    (N + block.x - 1) / block.x,
    (M + block.y - 1) / block.y
);
```

---

## 9. Main Experimental Findings

```text
1. The original 16×16 configuration is not universally optimal.

2. 16×16 remains best for very small workloads.

3. 32×8 becomes consistently better at SeqLen >= 512.

4. 32×8 reduces large-workload execution time by approximately 9–10%.

5. 8×8 performs poorly for large workloads.

6. 8×32 performs worst despite having the same 256 threads/block as 32×8.

7. Block shape matters independently of total thread count.

8. Mapping one 32-thread warp along the column dimension is favorable for the current MatMul memory access pattern.
```

---

## 10. Final Summary

Baseline:

```text
16×16
```

Best small-workload configuration:

```text
SeqLen <= 256
→ 16×16
```

Best medium/large-workload configuration:

```text
SeqLen >= 512
→ 32×8
```

Measured improvement for large workloads:

```text
~1.10× speedup
```

or approximately:

```text
9–10% lower kernel execution time
```

Strongest measured result:

```text
SeqLen = 1024

16×16:
0.103813 ms

32×8:
0.093778 ms

Speedup:
1.1070×
```

The main architectural explanation is:

```text
32×8
→ 32 threads in x dimension
→ one warp approximately maps to one output row
→ consecutive threads access consecutive B columns
→ more favorable coalesced memory behavior
```

The experimentally selected policy is therefore:

```text
SeqLen <= 256:
16×16

SeqLen >= 512:
32×8
```

This completes the empirical block-configuration tuning for the current naive MatMul kernel.



#
#
#
#
#
# AttentionScores Block Configuration Tuning Report

## 1. Objective

This experiment empirically tunes the CUDA block configuration for the current `AttentionScores` kernel.

Baseline:

```cpp
dim3 block(16, 16);
```

Because Nsight Compute hardware counters are unavailable in the current environment, optimization is performed by direct timing. The kernel math is unchanged; only block shape varies.

Tested configurations:

```text
16 × 16
8 × 8
32 × 8
8 × 32
```

---

## 2. Workload

The kernel computes scaled dot-product attention scores:

```text
Q:      [seq_len, 64]
K:      [seq_len, 64]
Scores: [seq_len, seq_len]
```

Core computation:

```cpp
for(int k = 0; k < head_dim; k++)
{
    sum += Q[row * head_dim + k]
         * K[col * head_dim + k];
}
```

with causal masking:

```cpp
if(col > row)
{
    scores[row * seq_len + col] = -INFINITY;
    return;
}
```

---

## 3. Experimental Setup

```text
Head Dimension: 64
Warmup Runs: 10
Benchmark Runs: 100
Sequence Lengths: 128, 256, 512, 1024, 2048
```

| Block | Threads/Block |
|---|---:|
| 16×16 | 256 |
| 8×8 | 64 |
| 32×8 | 256 |
| 8×32 | 256 |

All configurations passed correctness validation.

Relative performance:

```text
vs16x16 = baseline_time / tested_time
```

---

## 4. Results

### SeqLen = 128

| Block | Threads | Time (ms) | vs16x16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.009943 | 1.0000× | PASS |
| 8×8 | 64 | 0.007332 | 1.3561× | PASS |
| 32×8 | 256 | 0.016527 | 0.6016× | PASS |
| 8×32 | 256 | 0.007485 | 1.3283× | PASS |

Best:

```text
8×8
```

Time reduction:

```text
1 - 0.007332 / 0.009943 ≈ 26.26%
```

---

### SeqLen = 256

| Block | Threads | Time (ms) | vs16x16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.029379 | 1.0000× | PASS |
| 8×8 | 64 | 0.013517 | 2.1734× | PASS |
| 32×8 | 256 | 0.048691 | 0.6034× | PASS |
| 8×32 | 256 | 0.017428 | 1.6857× | PASS |

Best:

```text
8×8
```

Time reduction:

```text
1 - 0.013517 / 0.029379 ≈ 53.99%
```

This is the strongest measured improvement.

---

### SeqLen = 512

| Block | Threads | Time (ms) | vs16x16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.063559 | 1.0000× | PASS |
| 8×8 | 64 | 0.042404 | 1.4989× | PASS |
| 32×8 | 256 | 0.113910 | 0.5580× | PASS |
| 8×32 | 256 | 0.043284 | 1.4684× | PASS |

Best:

```text
8×8
```

Time reduction:

```text
1 - 0.042404 / 0.063559 ≈ 33.28%
```

---

### SeqLen = 1024

| Block | Threads | Time (ms) | vs16x16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.204390 | 1.0000× | PASS |
| 8×8 | 64 | 0.141301 | 1.4465× | PASS |
| 32×8 | 256 | 0.347116 | 0.5888× | PASS |
| 8×32 | 256 | 0.127212 | 1.6067× | PASS |

Best:

```text
8×32
```

Time reduction:

```text
1 - 0.127212 / 0.204390 ≈ 37.76%
```

---

### SeqLen = 2048

| Block | Threads | Time (ms) | vs16x16 | Correct |
|---|---:|---:|---:|---|
| 16×16 | 256 | 0.693862 | 1.0000× | PASS |
| 8×8 | 64 | 0.441559 | 1.5714× | PASS |
| 32×8 | 256 | 1.116969 | 0.6212× | PASS |
| 8×32 | 256 | 0.414249 | 1.6750× | PASS |

Best:

```text
8×32
```

Time reduction:

```text
1 - 0.414249 / 0.693862 ≈ 40.30%
```

---

## 5. Best Configuration Summary

| SeqLen | Best Block | Time (ms) | Speedup vs 16×16 |
|---:|---|---:|---:|
| 128 | 8×8 | 0.007332 | 1.3561× |
| 256 | 8×8 | 0.013517 | 2.1734× |
| 512 | 8×8 | 0.042404 | 1.4989× |
| 1024 | 8×32 | 0.127212 | 1.6067× |
| 2048 | 8×32 | 0.414249 | 1.6750× |

Recommended policy:

```text
SeqLen <= 512:
8×8

SeqLen >= 1024:
8×32
```

---

## 6. Why AttentionScores Behaves Differently from MatMul

The previous MatMul experiment favored `32×8` for large workloads.

AttentionScores shows the opposite result.

The reason is the access pattern.

MatMul uses:

```cpp
B[k * N + col]
```

Adjacent `col` values are adjacent in memory.

AttentionScores uses:

```cpp
K[col * head_dim + k]
```

with:

```text
head_dim = 64
```

Adjacent `col` values therefore refer to different K rows separated by:

```text
64 floats = 256 bytes
```

So a warp spanning 32 columns generates a highly strided access pattern.

---

## 7. Why 32×8 Performs Poorly

For:

```text
block = 32×8
```

one warp approximately maps to:

```text
1 row × 32 columns
```

For fixed `k`:

```text
T0  → K[0][k]
T1  → K[1][k]
...
T31 → K[31][k]
```

Each neighboring thread accesses memory about 256 bytes apart.

This is very different from the contiguous access pattern that benefited MatMul.

At SeqLen 2048:

```text
16×16 = 0.693862 ms
32×8  = 1.116969 ms
```

`32×8` is therefore substantially slower than the baseline and approximately:

```text
1.116969 / 0.414249 ≈ 2.70×
```

slower than the best `8×32` configuration.

---

## 8. Why x = 8 Performs Better

Both winning configurations use:

```text
blockDim.x = 8
```

They are:

```text
8×8
8×32
```

For `8×32`, one warp approximately spans:

```text
8 columns × 4 rows
```

Conceptually:

```text
Threads 0–7   → row 0, col 0–7
Threads 8–15  → row 1, col 0–7
Threads 16–23 → row 2, col 0–7
Threads 24–31 → row 3, col 0–7
```

For fixed `k`, the same eight K values:

```text
K[0][k] ... K[7][k]
```

are needed by several output rows.

This creates potential reuse of K values across threads working on different rows.

By contrast, `32×8` uses 32 distinct K rows in each warp and provides much less of this reuse.

The experiment strongly favors the smaller x dimension.

---

## 9. Why 8×8 Wins for Small and Medium Workloads

`8×8` uses:

```text
64 threads/block = 2 warps/block
```

It is best for:

```text
SeqLen = 128
SeqLen = 256
SeqLen = 512
```

Measured speedups:

```text
128  → 1.3561×
256  → 2.1734×
512  → 1.4989×
```

The smaller block can provide more flexible scheduling and enough parallel blocks for these workload sizes.

---

## 10. Why 8×32 Wins for Large Workloads

`8×32` preserves the favorable:

```text
blockDim.x = 8
```

while increasing the block size to:

```text
256 threads/block
```

For larger matrices, this becomes faster.

At SeqLen 1024:

```text
8×8  = 0.141301 ms
8×32 = 0.127212 ms
```

At SeqLen 2048:

```text
8×8  = 0.441559 ms
8×32 = 0.414249 ms
```

Therefore the workload-dependent transition is:

```text
small / medium:
8×8

large:
8×32
```

---

## 11. Comparison with Baseline

Selected best configuration vs original 16×16:

| SeqLen | Baseline | Best | Speedup | Time Reduction |
|---:|---:|---:|---:|---:|
| 128 | 0.009943 ms | 0.007332 ms | 1.3561× | 26.26% |
| 256 | 0.029379 ms | 0.013517 ms | 2.1734× | 53.99% |
| 512 | 0.063559 ms | 0.042404 ms | 1.4989× | 33.28% |
| 1024 | 0.204390 ms | 0.127212 ms | 1.6067× | 37.76% |
| 2048 | 0.693862 ms | 0.414249 ms | 1.6750× | 40.30% |

The original 16×16 configuration is never the fastest in this experiment.

---

## 12. Recommended Runtime Policy

Recommended launch configuration:

```cpp
dim3 block;

if(seq_len <= 512)
{
    block = dim3(8, 8);
}
else
{
    block = dim3(8, 32);
}
```

Grid:

```cpp
dim3 grid(
    (seq_len + block.x - 1) / block.x,
    (seq_len + block.y - 1) / block.y
);
```

---

## 13. Main Findings

```text
1. The original 16×16 configuration is not optimal.

2. 8×8 is best for SeqLen 128, 256, and 512.

3. 8×32 is best for SeqLen 1024 and 2048.

4. 32×8 is consistently the worst configuration.

5. Every best-performing configuration uses blockDim.x = 8.

6. The result is the opposite of MatMul.

7. This difference is explained by the K[col][k] memory access pattern.

8. Smaller x allows better reuse of K values across multiple output rows.

9. Large workloads benefit from increasing y while keeping x = 8.

10. The tuning improves execution time by roughly 26%–54% across tested workloads.
```

---

## 14. Final Summary

Baseline:

```text
16×16
```

Selected configuration:

```text
SeqLen <= 512:
8×8

SeqLen >= 1024:
8×32
```

Best measured speedup:

```text
SeqLen = 256
2.1734×
```

Largest time reduction:

```text
~54%
```

Large-workload results:

```text
SeqLen 1024:
1.6067× speedup

SeqLen 2048:
1.6750× speedup
```

Central architectural observation:

```text
MatMul:
32×8 is beneficial because neighboring column accesses are contiguous.

AttentionScores:
32×8 is harmful because neighboring column accesses to K are separated
by head_dim = 64 floats.

AttentionScores instead benefits from x = 8, allowing multiple output
rows in a warp to reuse a smaller set of K values.
```

This completes the empirical block-configuration tuning for the current `AttentionScores` CUDA kernel.



#
#
#
#
#
# AttentionSoftmax Block Size Tuning Report

## 1. Objective

This experiment empirically tunes the CUDA block size for the current `AttentionSoftmax` kernel.

The original implementation uses:

```cpp
const int block_size = 256;
```

The kernel algorithm is unchanged. Only the number of threads per block is varied.

Tested block sizes:

```text
64
128
256
512
```

The original `256 threads/block` configuration is used as the baseline.

---

## 2. Kernel Structure

Each CUDA block processes one row of the Attention score matrix:

```text
scores:  [seq_len, seq_len]
weights: [seq_len, seq_len]
```

Each row performs:

```text
1. row maximum reduction
2. exp(score - max)
3. row sum reduction
4. normalization
```

The row is distributed across threads using:

```cpp
for(int col = tid; col < seq_len; col += blockDim.x)
```

Therefore, block size directly controls how much work each thread performs.

---

## 3. Experimental Setup

```text
Warmup Runs: 10
Benchmark Runs: 100
Baseline Block Size: 256
```

Sequence lengths:

```text
128
256
512
1024
2048
```

All tested configurations passed correctness validation.

Relative performance:

```text
vs256 = baseline_time / tested_time
```

Therefore:

```text
> 1.0 = faster than 256
= 1.0 = equal to 256
< 1.0 = slower than 256
```

---

## 4. Experimental Results

### SeqLen = 128

| Block Size | Time (ms) | vs256 | Correct |
|---:|---:|---:|---|
| 64 | 0.004209 | 1.0171× | PASS |
| 128 | 0.003993 | 1.0720× | PASS |
| 256 | 0.004281 | 1.0000× | PASS |
| 512 | 0.005172 | 0.8277× | PASS |

Best:

```text
128 threads/block
```

Speedup:

```text
1.0720×
```

Time reduction:

```text
1 - 0.003993 / 0.004281
≈ 6.73%
```

---

### SeqLen = 256

| Block Size | Time (ms) | vs256 | Correct |
|---:|---:|---:|---|
| 64 | 0.005417 | 0.9299× | PASS |
| 128 | 0.004751 | 1.0603× | PASS |
| 256 | 0.005038 | 1.0000× | PASS |
| 512 | 0.007117 | 0.7079× | PASS |

Best:

```text
128 threads/block
```

Speedup:

```text
1.0603×
```

Time reduction:

```text
1 - 0.004751 / 0.005038
≈ 5.70%
```

---

### SeqLen = 512

| Block Size | Time (ms) | vs256 | Correct |
|---:|---:|---:|---|
| 64 | 0.008171 | 0.9662× | PASS |
| 128 | 0.006901 | 1.1439× | PASS |
| 256 | 0.007895 | 1.0000× | PASS |
| 512 | 0.011592 | 0.6810× | PASS |

Best:

```text
128 threads/block
```

Speedup:

```text
1.1439×
```

Time reduction:

```text
1 - 0.006901 / 0.007895
≈ 12.59%
```

This is the strongest improvement among the workloads where 128 is optimal.

---

### SeqLen = 1024

| Block Size | Time (ms) | vs256 | Correct |
|---:|---:|---:|---|
| 64 | 0.023020 | 0.8301× | PASS |
| 128 | 0.018032 | 1.0596× | PASS |
| 256 | 0.019108 | 1.0000× | PASS |
| 512 | 0.024730 | 0.7727× | PASS |

Best:

```text
128 threads/block
```

Speedup:

```text
1.0596×
```

Time reduction:

```text
1 - 0.018032 / 0.019108
≈ 5.63%
```

---

### SeqLen = 2048

| Block Size | Time (ms) | vs256 | Correct |
|---:|---:|---:|---|
| 64 | 0.117248 | 0.6245× | PASS |
| 128 | 0.101643 | 0.7204× | PASS |
| 256 | 0.073226 | 1.0000× | PASS |
| 512 | 0.067973 | 1.0773× | PASS |

Best:

```text
512 threads/block
```

Speedup:

```text
1.0773×
```

Time reduction:

```text
1 - 0.067973 / 0.073226
≈ 7.17%
```

This is the first tested workload where the larger 512-thread configuration becomes optimal.

---

## 5. Best Configuration Summary

| SeqLen | Best Block Size | Time (ms) | Speedup vs 256 |
|---:|---:|---:|---:|
| 128 | 128 | 0.003993 | 1.0720× |
| 256 | 128 | 0.004751 | 1.0603× |
| 512 | 128 | 0.006901 | 1.1439× |
| 1024 | 128 | 0.018032 | 1.0596× |
| 2048 | 512 | 0.067973 | 1.0773× |

Empirical policy:

```text
SeqLen <= 1024:
128 threads/block

SeqLen = 2048:
512 threads/block
```

---

## 6. Why 128 Threads Performs Best for Small and Medium Rows

For the first four sequence lengths, `128 threads/block` consistently outperforms the current 256-thread baseline.

The amount of work per thread is approximately:

```text
SeqLen 128:
128 threads → ~1 element/thread

SeqLen 256:
128 threads → ~2 elements/thread

SeqLen 512:
128 threads → ~4 elements/thread

SeqLen 1024:
128 threads → ~8 elements/thread
```

This provides enough parallelism while keeping the reduction tree relatively small.

For 128 threads, each reduction requires approximately:

```text
128
→ 64
→ 32
→ 16
→ 8
→ 4
→ 2
→ 1
```

This means fewer reduction stages and fewer synchronization rounds than a 256- or 512-thread block.

The measured data suggests that, up to SeqLen 1024, this balance between:

```text
parallel work
+
per-thread work
+
reduction overhead
+
synchronization overhead
```

is better than the larger configurations.

---

## 7. Why 512 Threads Performs Poorly for Small Rows

At small sequence lengths, 512 threads are excessive.

For example:

```text
SeqLen = 128
Block Size = 512
```

Only the first 128 threads have row elements to process during the initial data pass.

A large fraction of the block therefore contributes little or no useful row work while still participating in the reduction and synchronization structure.

The 512-thread reduction also requires more stages:

```text
512
→ 256
→ 128
→ 64
→ 32
→ 16
→ 8
→ 4
→ 2
→ 1
```

This adds synchronization overhead.

The results reflect this clearly.

At SeqLen 512:

```text
128 threads:
0.006901 ms

512 threads:
0.011592 ms
```

The 512-thread configuration is approximately:

```text
0.011592 / 0.006901
≈ 1.68×
```

slower than the best configuration.

---

## 8. Why 64 Threads Degrades as Sequence Length Grows

A 64-thread block has a smaller reduction tree, but each thread must process more elements.

Approximate work per thread:

```text
SeqLen 512:
~8 elements/thread

SeqLen 1024:
~16 elements/thread

SeqLen 2048:
~32 elements/thread
```

This increases the amount of serial work performed by each thread.

At SeqLen 2048:

```text
64 threads:
0.117248 ms

256 threads:
0.073226 ms

512 threads:
0.067973 ms
```

The 64-thread configuration is approximately:

```text
0.117248 / 0.067973
≈ 1.72×
```

slower than the best 512-thread configuration.

This shows that the reduced synchronization overhead of a small block is no longer enough to compensate for excessive per-thread work.

---

## 9. Why 512 Threads Becomes Best at SeqLen 2048

SeqLen 2048 changes the balance.

Work per thread becomes approximately:

```text
64 threads:
2048 / 64 = 32 elements/thread

128 threads:
2048 / 128 = 16 elements/thread

256 threads:
2048 / 256 = 8 elements/thread

512 threads:
2048 / 512 = 4 elements/thread
```

At this workload size, the row is large enough that additional parallelism becomes more valuable.

Although 512 threads require more reduction stages and synchronization, the reduction in serial work per thread is large enough to compensate.

Measured result:

```text
256 threads:
0.073226 ms

512 threads:
0.067973 ms
```

The 512-thread version is:

```text
1.0773×
```

faster.

This indicates a workload-dependent crossover where:

```text
additional parallelism
>
additional reduction/synchronization overhead
```

---

## 10. Important Crossover Behavior

The most interesting result is the sharp change between:

```text
SeqLen = 1024
```

and:

```text
SeqLen = 2048
```

At 1024:

```text
128 → 0.018032 ms
256 → 0.019108 ms
512 → 0.024730 ms
```

so 128 is best.

At 2048:

```text
128 → 0.101643 ms
256 → 0.073226 ms
512 → 0.067973 ms
```

so 512 is best.

This means the optimal configuration cannot be described by a single universal block size.

The kernel has at least two performance regimes:

```text
small / medium rows
→ lower block size is better

large rows
→ higher block size is better
```

---

## 11. Comparison with the Original 256-Thread Baseline

The current baseline is:

```text
256 threads/block
```

However, it is not the best configuration at any tested sequence length.

Selected best configuration vs baseline:

| SeqLen | Baseline 256 | Best | Speedup | Time Reduction |
|---:|---:|---:|---:|---:|
| 128 | 0.004281 | 0.003993 | 1.0720× | 6.73% |
| 256 | 0.005038 | 0.004751 | 1.0603× | 5.70% |
| 512 | 0.007895 | 0.006901 | 1.1439× | 12.59% |
| 1024 | 0.019108 | 0.018032 | 1.0596× | 5.63% |
| 2048 | 0.073226 | 0.067973 | 1.0773× | 7.17% |

Therefore, empirical tuning improves performance at every tested workload size.

---

## 12. Recommended Runtime Policy

Based strictly on the measured data:

```cpp
int block_size;

if(seq_len <= 1024)
{
    block_size = 128;
}
else
{
    block_size = 512;
}
```

For the currently tested sizes:

```text
128   → 128 threads
256   → 128 threads
512   → 128 threads
1024  → 128 threads
2048  → 512 threads
```

A more conservative implementation could explicitly use:

```cpp
int block_size =
    (seq_len >= 2048)
    ? 512
    : 128;
```

Since only SeqLen 2048 has currently demonstrated that 512 is superior, this threshold should be treated as an empirical policy for the tested workload range rather than a universal rule for every larger sequence length.

---

## 13. Main Findings

```text
1. The original 256-thread block is not optimal for any tested workload.

2. 128 threads/block is best for SeqLen 128, 256, 512, and 1024.

3. 512 threads/block becomes best at SeqLen 2048.

4. 64 threads/block becomes increasingly inefficient as sequence length grows.

5. 512 threads/block is inefficient for small and medium rows because of excess threads and larger reduction/synchronization cost.

6. At SeqLen 2048, increased parallelism outweighs the extra reduction cost.

7. The kernel exhibits a clear workload-dependent block-size crossover.

8. The best block size is determined by the balance between per-thread serial work and reduction/synchronization overhead.
```

---

## 14. Final Summary

Original baseline:

```text
256 threads/block
```

Best configuration for SeqLen <= 1024:

```text
128 threads/block
```

Best configuration at SeqLen 2048:

```text
512 threads/block
```

Best measured speedup over the original baseline:

```text
SeqLen = 512
1.1439×
```

Largest measured time reduction:

```text
~12.59%
```

The central performance tradeoff is:

```text
smaller block
→ less reduction/synchronization overhead
→ more work per thread

larger block
→ more reduction/synchronization overhead
→ less work per thread
```

For small and medium rows:

```text
128 threads
```

provides the best balance.

For a 2048-element row:

```text
512 threads
```

provides enough additional parallelism to overcome its larger reduction cost.

The experimentally selected policy for the tested workload range is:

```cpp
if(seq_len >= 2048)
{
    block_size = 512;
}
else
{
    block_size = 128;
}
```

This completes the empirical block-size tuning for the current single-block `AttentionSoftmax` CUDA kernel.
