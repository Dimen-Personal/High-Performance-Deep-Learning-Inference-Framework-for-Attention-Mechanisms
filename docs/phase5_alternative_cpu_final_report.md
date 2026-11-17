# Phase 5 Alternative — Final CPU Optimization Report

## 1. Overview

This report consolidates the complete CPU optimization work from Phase 5 Alternative.

It contains two major stages:

1. **Stage A — CPU Operator Fusion**
   - MatMul + Bias + GELU
   - Goal: remove intermediate tensors and reduce memory traffic.

2. **Stage B — CPU Cache-Aware Tiled Attention**
   - Query blocking
   - K/V blocking
   - Causal masking
   - Online softmax
   - No full `N x N` scores/weights
   - Goal: reduce quadratic intermediate memory and improve cache locality.

An important part of this investigation was that the **first Stage B benchmark appeared slower than the baseline**. We initially analyzed that result as a possible cost of tiling, explicit tile copies, online-softmax bookkeeping, and extra control flow.

However, after re-checking the build configuration, we found that the first Stage B benchmark had been run from the default `build/` directory, where the project default optimization level is:

```text
OPT_LEVEL = O0
```

Because the tiled implementation contains significantly more small loops, indexing operations, branches, and online-softmax state updates than the simple baseline, it is disproportionately penalized under O0.

We therefore rebuilt the CPU implementation with:

```bash
-DOPT_LEVEL=O3
```

and repeated both correctness and performance testing.

The final result is that **the cache-aware tiled CPU attention is both memory-efficient and significantly faster under O3**, achieving approximately:

```text
2.32x - 2.78x speedup
```

with the advantage increasing as sequence length grows.

---

# 2. Stage A — CPU MatMul + Bias + GELU Fusion

## 2.1 Baseline Design

The unfused path is:

```text
MatMul
  -> temp_matmul[M,N]

Bias Add
  -> temp_bias[M,N]

GELU
  -> output[M,N]
```

The fused path is:

```text
MatMul accumulator
  -> Bias
  -> GELU
  -> output[M,N]
```

The key difference is that the fused implementation does not materialize the two intermediate `M x N` tensors.

---

## 2.2 Stage A Correctness

All Stage A correctness tests passed with exactly zero numerical error.

| M | K | N | MAE | RMSE | Max Error | Result |
|---:|---:|---:|---:|---:|---:|:---|
| 1 | 64 | 64 | 0 | 0 | 0 | PASS |
| 16 | 64 | 64 | 0 | 0 | 0 | PASS |
| 32 | 128 | 128 | 0 | 0 | 0 | PASS |
| 64 | 256 | 256 | 0 | 0 | 0 | PASS |
| 128 | 256 | 512 | 0 | 0 | 0 | PASS |

This is expected because the fused implementation preserves the same arithmetic order for the MatMul accumulation and applies the same Bias and GELU formula.

---

# 3. Stage A — O3 Benchmark

The latest O3 results are:

| M | K | N | Unfused (ms) | Fused (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 128 | 0.532955 | 0.573918 | 0.9286x |
| 64 | 256 | 256 | 4.519951 | 4.772749 | 0.9470x |
| 128 | 256 | 512 | 18.771697 | 19.702865 | 0.9527x |
| 256 | 512 | 512 | 97.548276 | 98.728661 | 0.9880x |
| 512 | 512 | 512 | 195.327604 | 194.982043 | 1.0018x |
| 768 | 512 | 768 | 371.915885 | 389.579503 | 0.9547x |
| 1024 | 512 | 1024 | 1552.510252 | 1520.976115 | 1.0207x |

The values fluctuate around `1.0x`.

Therefore, Stage A should still be interpreted as:

> **Operator fusion substantially improves memory behavior, but end-to-end latency is approximately neutral because the scalar MatMul dominates execution time.**

At the largest tested shape:

```text
1024 x 512 x 1024
```

the fused implementation reaches:

```text
1.0207x
```

which is only a small latency improvement.

---

# 4. Stage A — Intermediate Memory Reduction

For an output tensor of shape `M x N`:

## Unfused

Temporary memory:

```text
temp_matmul[M,N]
temp_bias[M,N]
```

Total temporary storage:

```text
2 * M * N * sizeof(float)
```

Logical intermediate/output traffic:

```text
1. MatMul writes temp_matmul
2. Bias reads temp_matmul
3. Bias writes temp_bias
4. GELU reads temp_bias
5. GELU writes output
```

Approximate traffic:

```text
5 * M * N * sizeof(float)
```

## Fused

The fused implementation directly produces the final output:

```text
MatMul accumulator
-> Bias
-> GELU
-> output
```

Therefore:

```text
Temporary M x N tensors = 0
```

and logical intermediate/output traffic becomes:

```text
1 * M * N * sizeof(float)
```

This corresponds to an estimated:

```text
80% reduction
```

in intermediate/output traffic.

### Measured memory table

| M | K | N | Unfused Temp (MB) | Fused Temp (MB) | Unfused Traffic (MB) | Fused Traffic (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 128 | 128 | 0.0312 | 0.0000 | 0.0781 | 0.0156 |
| 64 | 256 | 256 | 0.1250 | 0.0000 | 0.3125 | 0.0625 |
| 128 | 256 | 512 | 0.5000 | 0.0000 | 1.2500 | 0.2500 |
| 256 | 512 | 512 | 1.0000 | 0.0000 | 2.5000 | 0.5000 |
| 512 | 512 | 512 | 2.0000 | 0.0000 | 5.0000 | 1.0000 |
| 768 | 512 | 768 | 4.5000 | 0.0000 | 11.2500 | 2.2500 |
| 1024 | 512 | 1024 | 8.0000 | 0.0000 | 20.0000 | 4.0000 |

---

# 5. Stage A — Cachegrind Analysis

For a `512 x 512 x 512` profile:

| Metric | Unfused | Fused | Change |
|---|---:|---:|---:|
| Ir | 989,553,382 | 985,405,975 | -0.42% |
| Dr | 279,979,286 | 280,369,669 | +0.14% |
| Dw | 8,076,167 | 5,651,158 | **-30.0%** |
| D1mr | 134,565,732 | 134,638,016 | ~0% |
| D1mw | 412,255 | 360,292 | **-12.6%** |
| DLmr | 7,713 | 7,713 | 0% |
| DLmw | 83,556 | 50,786 | **-39.2%** |

Definitions:

```text
Ir    instruction reads
Dr    data reads
Dw    data writes

D1mr  L1 data read misses
D1mw  L1 data write misses

DLmr  last-level data read misses
DLmw  last-level data write misses
```

The most significant improvement is in the write path:

```text
Data writes:
-30.0%

L1 data write misses:
-12.6%

Last-level data write misses:
-39.2%
```

However, read misses remain effectively unchanged.

The function-level profile explains why.

The original scalar `matmul()` accounts for approximately:

```text
95.24% of instructions
95.88% of data reads
99.97% of L1 data read misses
```

Therefore, removing Bias/GELU intermediate tensors does not significantly change the dominant read-side bottleneck.

This supports the Stage A conclusion:

> Fusion successfully reduces memory traffic, but the scalar MatMul remains the dominant compute and cache-read bottleneck.

---

# 6. Stage B — CPU Cache-Aware Tiled Attention

## 6.1 Original Baseline

The baseline CPU attention path is:

```text
Q K^T
  ->
scores[N,N]

Softmax
  ->
weights[N,N]

weights x V
  ->
output[N,d]
```

This requires two full quadratic intermediate tensors:

```text
scores[N,N]
weights[N,N]
```

Temporary memory:

```text
2 * N^2 * sizeof(float)
```

---

# 7. Stage B — Cache-Aware Design

The tiled implementation uses:

```text
Query Block Size = 16
Key Block Size   = 32
Head Dimension   = 64
```

The execution pattern is:

```text
Q block
   x
K/V block
   ->
small score tile
   ->
online softmax
   ->
direct weighted-V accumulation
   ->
next K/V block
```

The tiled version does not materialize:

```text
scores[N,N]
weights[N,N]
```

Instead, the working set consists only of:

```text
q_tile
k_tile
v_tile
score_tile
row_max
row_sum
output_accumulator
```

For the selected configuration:

```text
BQ = 16
BK = 32
d  = 64
```

the approximate tiled working set is:

```text
26.125 KB
```

and is essentially independent of sequence length.

---

# 8. Stage B — Correctness

All cache-aware tiled attention tests passed.

The test intentionally includes boundary-sensitive sizes around both block dimensions:

```text
Query block boundary:
15, 16, 17

Key block boundary:
31, 32, 33

Additional:
63, 64, 65
```

Results:

| SeqLen | MAE | RMSE | Max Error | Result |
|---:|---:|---:|---:|:---|
| 1 | 0 | 0 | 0 | PASS |
| 2 | 3.063e-09 | 7.259e-09 | 2.980e-08 | PASS |
| 15 | 5.286e-09 | 7.895e-09 | 2.980e-08 | PASS |
| 16 | 5.330e-09 | 7.898e-09 | 2.980e-08 | PASS |
| 17 | 5.283e-09 | 7.802e-09 | 2.980e-08 | PASS |
| 31 | 5.107e-09 | 7.483e-09 | 2.980e-08 | PASS |
| 32 | 5.165e-09 | 7.563e-09 | 2.980e-08 | PASS |
| 33 | 5.136e-09 | 7.506e-09 | 2.980e-08 | PASS |
| 63 | 6.180e-09 | 9.015e-09 | 5.960e-08 | PASS |
| 64 | 6.211e-09 | 9.049e-09 | 5.960e-08 | PASS |
| 65 | 6.306e-09 | 9.189e-09 | 5.960e-08 | PASS |
| 128 | 8.089e-09 | 1.200e-08 | 8.941e-08 | PASS |
| 256 | 1.060e-08 | 1.580e-08 | 1.192e-07 | PASS |
| 512 | 1.580e-08 | 2.472e-08 | 2.682e-07 | PASS |

These tiny differences are expected because the tiled online-softmax implementation changes floating-point accumulation order.

---

# 9. Initial Stage B Result — O0 Investigation

The first benchmark produced the following unexpected result:

| SeqLen | Baseline (ms) | Tiled (ms) | Speedup |
|---:|---:|---:|---:|
| 128 | 4.682720 | 7.090562 | 0.6604x |
| 256 | 19.142865 | 27.716911 | 0.6907x |
| 512 | 76.307303 | 110.072011 | 0.6932x |
| 1024 | 310.526422 | 436.938869 | 0.7107x |

At this point, the tiled implementation appeared approximately:

```text
29% - 34% slower
```

than the baseline.

The initial interpretation was that the tiled algorithm might be paying too much overhead from:

- explicit Q/K/V tile copies
- more nested loops
- more index calculations
- per-tile boundary checks
- online-softmax state updates
- repeated output-accumulator rescaling
- additional `exp()` operations

This interpretation was reasonable based on the observed result.

However, the result conflicted with the expected benefit from:

- removing `N x N` intermediate tensors
- processing only causal-valid keys
- improving K/V reuse
- reducing memory round-trips

Therefore the benchmark configuration was re-checked.

---

# 10. Root Cause — O0 Build

The project's CMake default is:

```text
OPT_LEVEL = O0
```

The initial Stage B benchmark was executed from:

```bash
./build/cpu_fused_matmul_bias_gelu_benchmark
```

rather than the explicitly optimized:

```bash
./build-O3/cpu_fused_matmul_bias_gelu_benchmark
```

This was confirmed indirectly by Stage A timing.

For example, at:

```text
1024 x 512 x 1024
```

the O0-style result was approximately:

```text
~3.4 seconds
```

while the O3 result is approximately:

```text
~1.5 seconds
```

This large difference strongly indicates that the original Stage B benchmark was not representative of optimized CPU execution.

---

# 11. Why O0 Hurts Tiled Attention More

The baseline attention implementation contains relatively simple loops:

```text
attention_scores:
for i
  for j
    for k

softmax:
max loop
exp loop
normalize loop

matmul:
for i
  for j
    for k
```

The tiled implementation introduces much more control structure:

```text
query-block loop
  key-block loop
    valid-key calculation
    tile copies
    score-tile loops
    tile max
    online max merge
    alpha calculation
    output rescaling
    beta calculation
    weighted-V accumulation
```

Under O0, the compiler does very little optimization of:

- repeated index arithmetic
- loop-invariant expressions
- inlining
- branch simplification
- strength reduction
- vectorization
- common-subexpression elimination
- constant propagation

Therefore, the tiled implementation suffers much more than the simpler baseline.

This led to a false performance conclusion.

---

# 12. Final Stage B O3 Benchmark

After rebuilding with:

```bash
cmake -S . -B build-O3 -DOPT_LEVEL=O3
```

the performance result changed dramatically.

| SeqLen | Baseline (ms) | Tiled (ms) | Speedup |
|---:|---:|---:|---:|
| 128 | 1.439526 | 0.620039 | **2.3217x** |
| 256 | 6.114154 | 2.461027 | **2.4844x** |
| 512 | 25.232947 | 9.807633 | **2.5728x** |
| 1024 | 108.666167 | 39.115329 | **2.7781x** |

This completely reverses the O0 result.

The optimized tiled attention is now:

```text
2.32x faster at N=128
2.48x faster at N=256
2.57x faster at N=512
2.78x faster at N=1024
```

---

# 13. O0 vs O3 Comparison

## Baseline

| SeqLen | O0 Baseline (ms) | O3 Baseline (ms) | O3 Improvement |
|---:|---:|---:|---:|
| 128 | 4.682720 | 1.439526 | 3.25x faster |
| 256 | 19.142865 | 6.114154 | 3.13x faster |
| 512 | 76.307303 | 25.232947 | 3.02x faster |
| 1024 | 310.526422 | 108.666167 | 2.86x faster |

## Tiled

| SeqLen | O0 Tiled (ms) | O3 Tiled (ms) | O3 Improvement |
|---:|---:|---:|---:|
| 128 | 7.090562 | 0.620039 | 11.44x faster |
| 256 | 27.716911 | 2.461027 | 11.26x faster |
| 512 | 110.072011 | 9.807633 | 11.22x faster |
| 1024 | 436.938869 | 39.115329 | 11.17x faster |

This is one of the strongest observations in the whole CPU investigation.

The baseline improves by roughly:

```text
2.9x - 3.3x
```

under O3.

The tiled implementation improves by roughly:

```text
11.2x - 11.4x
```

under O3.

This shows that the tiled implementation is **far more dependent on compiler optimization** because of its additional loop structure and bookkeeping.

---

# 14. Stage B Memory Scaling

The baseline requires:

```text
scores[N,N]
weights[N,N]
```

Therefore temporary storage grows quadratically:

```text
O(N^2)
```

Measured values:

| SeqLen | Baseline Temp (MB) | Tiled Working Set (KB) |
|---:|---:|---:|
| 128 | 0.125 | 26.125 |
| 256 | 0.500 | 26.125 |
| 512 | 2.000 | 26.125 |
| 1024 | 8.000 | 26.125 |

The tiled working set remains effectively constant:

```text
26.125 KB
```

because it depends on the fixed block sizes rather than sequence length.

This means the tiled implementation achieves:

```text
Baseline temporary memory:
O(N^2)

Tiled temporary memory:
O(Bq*d + Bk*d + Bq*Bk)
```

For fixed block sizes, this is effectively constant with respect to `N`.

---

# 15. Why O3 Tiled Attention Is Faster

The final O3 result matches the intended algorithmic design.

Several factors contribute.

## 15.1 No Full N x N Intermediate Materialization

The baseline must:

```text
write scores[N,N]
read scores[N,N]

write weights[N,N]
read weights[N,N]
```

The tiled implementation avoids these full tensors.

This removes a large amount of intermediate memory traffic.

---

## 15.2 Causal Work Reduction

The baseline score stage avoids QK compute above the diagonal, but the final baseline MatMul still processes the entire:

```text
weights[N,N] x V[N,d]
```

matrix product.

Even masked weights equal to zero are still included in the MatMul loops.

The tiled implementation instead processes only:

```text
key <= query
```

for the weighted-V accumulation.

Therefore it avoids a significant fraction of useless work.

---

## 15.3 K/V Tile Reuse

For a query block:

```text
BQ = 16
```

the same K/V tile can be reused across multiple queries.

This improves locality and reduces repeated long-distance memory accesses.

---

## 15.4 Small Working Set

At:

```text
BQ = 16
BK = 32
d = 64
```

the total working set is approximately:

```text
26.125 KB
```

which is small enough to fit near a typical 32 KB L1 data-cache capacity.

This strongly improves temporal locality.

---

## 15.5 Online Softmax Eliminates the Weight Matrix

The tiled implementation performs:

```text
score calculation
+
softmax state update
+
weighted V accumulation
```

without materializing the full normalized attention matrix.

This is the CPU analogue of the core FlashAttention idea.

---

# 16. Speedup Trend

The speedup increases with sequence length:

```text
N=128  -> 2.3217x
N=256  -> 2.4844x
N=512  -> 2.5728x
N=1024 -> 2.7781x
```

This trend is important.

As `N` grows, the baseline's quadratic intermediate-memory cost becomes increasingly expensive.

The tiled implementation's temporary working set remains fixed.

Therefore the benefit of:

- reduced intermediate traffic
- better locality
- causal work reduction

becomes more important at larger sequence lengths.

This matches the theoretical expectation.

---

# 17. Combined Stage A and Stage B Interpretation

The two CPU experiments demonstrate two different optimization regimes.

## Stage A

```text
MatMul + Bias + GELU Fusion
```

reduces memory traffic, but the workload remains dominated by the original scalar MatMul.

Result:

```text
Memory behavior:
significantly improved

Latency:
approximately unchanged
```

## Stage B

```text
Cache-Aware Tiled Attention
```

changes the structure of the dominant computation itself.

It reduces:

- intermediate memory
- unnecessary causal work
- long-range memory traffic

while improving reuse.

Under O3, the compiler is able to optimize the more complex tiled loop structure effectively.

Result:

```text
Memory:
dramatically improved

Latency:
2.32x - 2.78x faster
```

---

# 18. Important Engineering Lesson

The O0-to-O3 investigation is itself an important result.

The same algorithm can appear slower or much faster depending on compiler optimization.

The initial benchmark suggested:

```text
Tiled Attention < Baseline
```

but this was not an accurate representation of optimized CPU execution.

After O3:

```text
Tiled Attention >> Baseline
```

The key lesson is:

> Performance conclusions for loop-heavy CPU kernels should not be made from unoptimized O0 builds.

This is especially true for algorithms containing:

- nested tiling loops
- many small helper expressions
- repeated index calculations
- online state updates
- branch-heavy boundary handling

Compiler optimization is an essential part of the final execution strategy.

---

# 19. Final CPU Results

## Stage A — Operator Fusion

Correctness:

```text
PASS
```

Memory improvements:

```text
Temporary tensors:
2 -> 0

Modeled intermediate/output traffic:
5MN -> MN

Data writes:
~30% lower

L1 write misses:
~12.6% lower

Last-level write misses:
~39.2% lower
```

Latency:

```text
approximately neutral
```

because scalar MatMul remains dominant.

---

## Stage B — Cache-Aware Tiled Attention

Correctness:

```text
PASS
```

Memory:

```text
Baseline:
O(N^2)

Tiled:
~26.125 KB fixed working set
```

Performance under O3:

```text
N=128  -> 2.3217x
N=256  -> 2.4844x
N=512  -> 2.5728x
N=1024 -> 2.7781x
```

The speedup grows with sequence length.

---

# 20. Final Conclusion

The CPU Phase 5 Alternative optimization is successful.

Stage A demonstrates that operator fusion can substantially reduce memory traffic even when latency remains limited by a dominant MatMul kernel.

Stage B demonstrates the stronger optimization opportunity:

> **Cache-aware tiled attention with online softmax eliminates quadratic intermediate storage and achieves a 2.32x to 2.78x speedup under O3.**

The initial O0 result incorrectly suggested that the tiled implementation was slower. Rebuilding with O3 revealed that the tiled algorithm is highly sensitive to compiler optimization and that its intended memory-locality advantages are realized only in an optimized build.

The final CPU design therefore achieves both:

```text
lower memory complexity
+
lower latency
```

and becomes increasingly advantageous as sequence length grows.

---

# 21. Final Status

```text
Stage A:
COMPLETE

Stage B:
COMPLETE

CPU Phase 5 Alternative:
COMPLETE
```

Recommended final benchmark configuration:

```text
Compiler:
-O3

Attention:
Query Block = 16
Key Block   = 32
Head Dim    = 64
```

The O3 configuration should be treated as the authoritative performance baseline for all future CPU comparisons.
