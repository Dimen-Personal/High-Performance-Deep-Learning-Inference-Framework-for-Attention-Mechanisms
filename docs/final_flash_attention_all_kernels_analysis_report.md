# Final Flash Attention Optimization Report

## 1. Objective

This report summarizes the complete Flash Attention optimization sequence implemented in the CUDA attention framework and compares every kernel developed during the project.

The benchmark compares five implementations:

1. **Baseline Attention**
2. **Naive Fused Attention**
3. **Original Tiled Flash Attention**
4. **Warp-Optimized Tiled Flash Attention**
5. **Multi-Query Optimized Flash Attention**

Benchmark configuration:

```text
Head Dimension: 64
Warmup Runs: 10
Benchmark Runs: 100
GPU Architecture Target: Volta / sm_70
```

The optimization goals were:

- eliminate full `N × N` attention-score and attention-weight materialization;
- introduce online softmax;
- improve QK parallelism;
- reduce synchronization overhead;
- reuse K/V tiles across multiple queries;
- determine whether an IO-aware Flash-style implementation can approach the original three-kernel baseline.

---

## 2. Raw Performance Results

| SeqLen | Baseline (ms) | Naive Fused (ms) | Original Tiled (ms) | Warp Optimized (ms) | Multi-Query (ms) |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.022139 | 0.039936 | 0.046347 | 0.036864 | 0.067768 |
| 256 | 0.048180 | 0.078039 | 0.090614 | 0.070676 | 0.134031 |
| 512 | 0.100198 | 0.181166 | 0.245340 | 0.160317 | 0.246651 |
| 1024 | 0.299151 | 0.605194 | 0.671068 | 0.425144 | 0.452250 |
| 2048 | 0.960707 | 1.884293 | 2.194934 | 1.371945 | 1.076111 |

---

## 3. Speedup Results

| SeqLen | Naive/Base | Tiled/Base | WarpOpt/Base | MultiQ/Base | Opt vs Tiled | Multi vs Warp |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.5544× | 0.4777× | 0.6006× | 0.3267× | 1.2572× | 0.5440× |
| 256 | 0.6174× | 0.5317× | 0.6817× | 0.3595× | 1.2821× | 0.5273× |
| 512 | 0.5531× | 0.4084× | 0.6250× | 0.4062× | 1.5303× | 0.6500× |
| 1024 | 0.4943× | 0.4458× | 0.7036× | 0.6615× | 1.5784× | 0.9401× |
| 2048 | 0.5098× | 0.4377× | 0.7003× | 0.8928× | 1.5999× | 1.2749× |

---

## 4. Main Result

The optimization sequence produced two distinct best-performing Flash-style kernels depending on sequence length.

For `seq_len = 128` through `1024`, the **Warp-Optimized Tiled** implementation is the fastest Flash-style kernel.

At `seq_len = 2048`, the **Multi-Query Optimized** implementation becomes the fastest Flash-style kernel:

```text
Warp Optimized:
1.371945 ms

Multi-Query:
1.076111 ms
```

This corresponds to:

```text
1.2749× speedup
~21.6% lower latency
```

relative to Warp Optimized.

Compared with the baseline at `2048`:

```text
Baseline:
0.960707 ms

Multi-Query:
1.076111 ms
```

the Multi-Query kernel is only about:

```text
12.0% slower
```

while avoiding the baseline's full `N × N` score and weight intermediate tensors.

---

# 5. Kernel-by-Kernel Analysis

## 5.1 Baseline Attention

### Structure

```text
Q, K
  ↓
AttentionScores
  ↓
scores[N,N] in global memory
  ↓
AttentionSoftmax
  ↓
weights[N,N] in global memory
  ↓
MatMul(weights, V)
  ↓
Output
```

### Strengths

The baseline is the fastest implementation at every tested sequence length.

Its major advantage is highly exposed CUDA parallelism:

- the score kernel uses a 2D grid;
- many query/key pairs are calculated simultaneously;
- softmax uses dedicated row-level blocks;
- the final matrix multiplication exposes another large 2D parallel workload.

The GPU therefore receives many independent blocks and threads.

### Weaknesses

The baseline explicitly materializes:

```text
scores[N,N]
weights[N,N]
```

For FP32, temporary memory is:

```text
2 × N² × 4 bytes
```

At `N = 2048` this reaches:

```text
32 MB
```

of temporary global memory.

The simplified logical score/weight intermediate traffic used by this benchmark reaches approximately:

```text
96 MB
```

at `N = 2048`.

This is a logical materialization estimate, not a measurement of physical HBM traffic.

### Interpretation

The baseline demonstrates that strong execution parallelism can still beat a more memory-efficient algorithm at moderate problem sizes.

---

## 5.2 Naive Fused Attention

### Structure

The naive fused kernel combines:

```text
scaled QK^T
+
causal mask
+
softmax
+
softmax × V
```

inside a single kernel.

It avoids materializing the full global-memory `N × N` score and weight matrices.

### Strengths

- kernel fusion;
- eliminates large global intermediates;
- fewer kernel launches;
- correct causal-attention computation.

### Weaknesses

Its score-row storage still grows with sequence length.

Conceptually:

```text
one block
=
one query
=
one full attention row
```

So it is fused, but it does not yet implement fixed-size K/V tiling.

Measured latency:

```text
128   0.039936 ms
256   0.078039 ms
512   0.181166 ms
1024  0.605194 ms
2048  1.884293 ms
```

It is slower than baseline across all tested sizes.

### Interpretation

Kernel fusion alone is not sufficient for high performance.

Removing global intermediates helps memory behavior, but efficient thread mapping and data reuse are equally important.

---

## 5.3 Original Tiled Flash Attention

### Main Features

This version introduced:

```text
TILE_K = 32
K/V shared-memory tiling
online softmax
causal tile-by-tile processing
fixed-size score storage
```

The online-softmax state stores only:

```text
running maximum
running normalization sum
running output numerator
```

rather than a complete attention row.

### Main Bottleneck

Its original QK mapping was:

```text
1 thread
→ 1 complete Q·K dot product
→ serial loop across head_dim
```

With:

```text
BLOCK_SIZE = 128
TILE_K = 32
head_dim = 64
```

only a fraction of the block performs useful score work at one time.

The implementation also contains several shared-memory reductions and repeated block-wide barriers.

### Measured Performance

```text
128   0.046347 ms
256   0.090614 ms
512   0.245340 ms
1024  0.671068 ms
2048  2.194934 ms
```

This is the slowest or near-slowest version across most test sizes.

### Interpretation

This kernel is algorithmically more Flash-like, but its execution mapping is inefficient.

The result demonstrates:

> Better asymptotic memory behavior does not automatically produce a faster GPU kernel.

---

## 5.4 Warp-Optimized Tiled Flash Attention

### Main Changes

The core score computation changed from:

```text
1 thread → 1 Q·K dot product
```

to:

```text
1 warp → 1 Q·K dot product
```

For `head_dim = 64`:

```text
32 lanes
×
about 2 dimensions per lane
```

Partial sums are combined using:

```text
__shfl_down_sync()
```

Additional improvements:

- Q-row caching;
- warp-level max reduction;
- warp-level sum reduction;
- removal of the shared reduction array;
- fewer block-wide synchronizations.

### Improvement over Original Tiled

| SeqLen | Original Tiled | Warp Optimized | Speedup |
|---:|---:|---:|---:|
| 128 | 0.046347 | 0.036864 | 1.2572× |
| 256 | 0.090614 | 0.070676 | 1.2821× |
| 512 | 0.245340 | 0.160317 | 1.5303× |
| 1024 | 0.671068 | 0.425144 | 1.5784× |
| 2048 | 2.194934 | 1.371945 | 1.5999× |

At `2048`, latency is reduced by approximately:

```text
37.5%
```

relative to Original Tiled.

### Interpretation

This confirms that:

```text
thread utilization
warp-level reduction
synchronization overhead
```

were major bottlenecks in the first tiled implementation.

This kernel is currently the best Flash-style implementation for short and medium sequence lengths.

---

## 5.5 Multi-Query Optimized Flash Attention

### Architectural Change

This version changes:

```text
1 block = 1 query
```

into:

```text
1 block = 4 queries
1 warp  = 1 query
```

Mapping:

```text
Warp 0 → Query q
Warp 1 → Query q+1
Warp 2 → Query q+2
Warp 3 → Query q+3
```

All four query warps reuse the same K/V tile loaded into shared memory.

### Expected Benefit

The main benefit is reduced repeated K/V global-memory traffic.

Instead of approximately:

```text
Query 0 block → load K/V tile
Query 1 block → load same K/V tile
Query 2 block → load same K/V tile
Query 3 block → load same K/V tile
```

the block performs:

```text
load K/V tile once
↓
reuse for four queries
```

### Why It Is Slower at Small Sequence Lengths

The optimization trades parallelism for reuse.

Compared with Warp Optimized:

- there are fewer blocks;
- one warp is permanently assigned to one query;
- each warp iterates over multiple keys;
- score-level parallelism is reduced;
- the block holds more per-query state.

Measured comparison:

| SeqLen | Warp Optimized | Multi-Query | Multi vs Warp |
|---:|---:|---:|---:|
| 128 | 0.036864 ms | 0.067768 ms | 0.5440× |
| 256 | 0.070676 ms | 0.134031 ms | 0.5273× |
| 512 | 0.160317 ms | 0.246651 ms | 0.6500× |
| 1024 | 0.425144 ms | 0.452250 ms | 0.9401× |
| 2048 | 1.371945 ms | 1.076111 ms | 1.2749× |

At short sequence lengths the cost of reduced parallelism dominates.

For example at `128`:

```text
Warp Optimized:
0.036864 ms

Multi-Query:
0.067768 ms
```

Multi-Query is approximately:

```text
83.8% slower
```

than Warp Optimized.

### Crossover Behavior

At `1024`:

```text
Warp Optimized:
0.425144 ms

Multi-Query:
0.452250 ms
```

The difference is only about:

```text
6.4%
```

At `2048`:

```text
Warp Optimized:
1.371945 ms

Multi-Query:
1.076111 ms
```

Multi-Query becomes faster by:

```text
1.2749×
```

or approximately:

```text
21.6% lower latency
```

### Interpretation

This is strong evidence that K/V reuse becomes more valuable as sequence length grows.

The trend is:

```text
short sequence
→ parallelism dominates

long sequence
→ memory reuse becomes increasingly important
```

The Multi-Query design therefore validates the core IO-aware optimization idea.

---

# 6. Best Flash-Style Kernel by Sequence Length

| SeqLen | Best Flash-Style Kernel | Latency |
|---:|---|---:|
| 128 | Warp Optimized | 0.036864 ms |
| 256 | Warp Optimized | 0.070676 ms |
| 512 | Warp Optimized | 0.160317 ms |
| 1024 | Warp Optimized | 0.425144 ms |
| 2048 | Multi-Query | 1.076111 ms |

The data suggests sequence-length-dependent dispatch may be beneficial:

```text
short / medium sequence
    → Warp Optimized

long sequence
    → Multi-Query
```

The exact crossover should be measured rather than assumed.

Useful additional test sizes:

```text
1152
1280
1536
1792
```

---

# 7. Memory Analysis

Baseline temporary memory:

| SeqLen | Temporary Memory |
|---:|---:|
| 128 | 0.125 MB |
| 256 | 0.500 MB |
| 512 | 2.000 MB |
| 1024 | 8.000 MB |
| 2048 | 32.000 MB |

Simplified logical intermediate traffic:

| SeqLen | Logical Intermediate Traffic |
|---:|---:|
| 128 | 0.375 MB |
| 256 | 1.500 MB |
| 512 | 6.000 MB |
| 1024 | 24.000 MB |
| 2048 | 96.000 MB |

The Flash-style kernels avoid the full global `scores[N,N]` and `weights[N,N]` intermediates.

This does **not** mean they have zero memory traffic. Q/K/V still reside in global memory and the final output is written globally.

The key memory advantage is:

```text
No full N × N score/weight materialization
```

which becomes increasingly important as `N` grows.

---

# 8. Optimization Evolution

## Stage 1 — Baseline

```text
High parallelism
Simple implementation
Large N×N intermediates
```

## Stage 2 — Naive Fusion

```text
No full global intermediates
But per-query score storage still grows with sequence length
```

## Stage 3 — Tiled Online Softmax

```text
Fixed tile storage
Online softmax
Correct Flash-style algorithmic structure
But poor thread mapping
```

## Stage 4 — Warp Optimization

```text
Warp-cooperative QK
Warp reductions
Fewer barriers
Much better compute utilization
```

## Stage 5 — Query Tiling / Multi-Query

```text
Multiple queries per CTA
K/V tile reuse
Better long-sequence IO behavior
```

This progression shows that high-performance attention requires both:

```text
IO-aware algorithm design
+
efficient GPU execution mapping
```

---

# 9. Current Ranking at N = 2048

```text
1. Baseline
   0.960707 ms

2. Multi-Query Optimized
   1.076111 ms

3. Warp-Optimized Tiled
   1.371945 ms

4. Naive Fused
   1.884293 ms

5. Original Tiled
   2.194934 ms
```

The Multi-Query kernel is only about:

```text
12.0%
```

slower than baseline.

At the same time, it avoids the baseline's:

```text
32 MB
```

of full `N × N` FP32 temporary score/weight storage.

This is the strongest overall Flash Attention result produced by the project so far.

---

# 10. Recommended Next Optimizations

## 10.1 Tune Queries Per Block

Current:

```text
4 queries per block
```

Test:

```text
2
4
8
```

A 2-query configuration may preserve more CTA-level parallelism while still benefiting from K/V reuse.

## 10.2 Locate the Crossover Precisely

Benchmark:

```text
1152
1280
1536
1792
```

to determine where Multi-Query first becomes faster than Warp Optimized.

## 10.3 Tune TILE_K

Current:

```text
TILE_K = 32
```

Test:

```text
16
32
64
```

only after preserving the current architecture so that each change can be measured independently.

## 10.4 Vectorized Global Loads

Evaluate aligned:

```text
float4
```

loads for K/V tiles.

## 10.5 Overlap Memory Movement and Compute

Potential future techniques:

```text
double buffering
warp specialization
software pipelining
```

The current V100/Volta target does not provide the same `cp.async` path available on Ampere and newer architectures.

## 10.6 FP16 / Tensor Cores

The current kernels are FP32.

A mixed-precision path could substantially increase QK and output throughput.

This should be treated as a separate optimization stage because it changes both numerical precision and execution units.

---

# 11. Main Technical Lessons

### 1. Kernel fusion is not automatically faster

The Naive Fused kernel removes global intermediates but remains slower than baseline.

### 2. Better memory complexity is not enough

Original Tiled has the right algorithmic structure but poor CUDA execution mapping.

### 3. Warp-level mapping matters

Warp cooperative QK and shuffle reductions improve Original Tiled by up to roughly:

```text
1.60×
```

### 4. Reuse competes with parallelism

Multi-Query is much slower for small sequences but faster at `2048`.

### 5. The optimal kernel depends on workload size

The current results do not support one universal winner for all sequence lengths.

### 6. Flash Attention's advantage becomes more important as N grows

Avoiding `N × N` intermediates and reusing K/V becomes increasingly valuable at long sequence lengths.

---

# 12. Final Conclusion

The project successfully progressed from basic three-kernel attention to an IO-aware tiled Flash Attention implementation.

The most important compute-side improvement was:

```text
warp-cooperative QK
+
warp-level reductions
```

The most important memory-side improvement was:

```text
multi-query K/V tile reuse
```

At `seq_len = 2048`, the best Flash-style implementation achieves:

```text
1.076111 ms
```

versus:

```text
0.960707 ms
```

for baseline.

The remaining latency gap is only about:

```text
12.0%
```

while the Flash implementation avoids the baseline's `32 MB` of full `N × N` temporary attention matrices.

The current result therefore demonstrates:

> An optimized Flash-style attention kernel can approach baseline latency while providing substantially better intermediate-memory scaling.

The next optimization phase should focus on balancing query reuse and GPU parallelism through parameter tuning, followed by vectorized loads and mixed-precision/Tensor-Core acceleration.
