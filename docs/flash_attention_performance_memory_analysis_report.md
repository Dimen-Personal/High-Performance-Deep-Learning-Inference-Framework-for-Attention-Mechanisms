# Flash Attention Performance & Memory Analysis Report

## 1. Objective

This benchmark compares three CUDA attention paths:

1. **Baseline Attention**: `AttentionScores -> AttentionSoftmax -> MatMul(weights, V)`
2. **Naive Fused Attention**: one fused kernel, one full score row in shared memory
3. **Tiled Flash Attention**: tiled K/V loading + online softmax

Configuration:

```text
Head Dimension: 64
Warmup Runs: 10
Benchmark Runs: 100
```

Tested sequence lengths:

```text
128, 256, 512, 1024, 2048
```

---

## 2. Raw Performance Results

| SeqLen | Baseline (ms) | Naive Fused (ms) | Tiled Flash (ms) | Naive Speedup | Flash Speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 0.021955 | 0.039485 | 0.045875 | 0.5560× | 0.4786× |
| 256 | 0.047626 | 0.077496 | 0.089948 | 0.6146× | 0.5295× |
| 512 | 0.100106 | 0.181187 | 0.245883 | 0.5525× | 0.4071× |
| 1024 | 0.316211 | 0.605706 | 0.715356 | 0.5221× | 0.4420× |
| 2048 | 0.999916 | 1.910999 | 2.193050 | 0.5232× | 0.4559× |

`Speedup = Baseline / Optimized`, so values below `1.0×` mean the optimized path is currently slower.

---

## 3. Intermediate Memory / Traffic

| SeqLen | Temp Memory (MB) | Min Intermediate Traffic (MB) | Current Baseline Logical Traffic (MB) |
|---:|---:|---:|---:|
| 128 | 0.1250 | 0.2500 | 0.3750 |
| 256 | 0.5000 | 1.0000 | 1.5000 |
| 512 | 2.0000 | 4.0000 | 6.0000 |
| 1024 | 8.0000 | 16.0000 | 24.0000 |
| 2048 | 32.0000 | 64.0000 | 96.0000 |

The baseline materializes:

```text
scores[N,N] + weights[N,N]
```

so temporary global memory is `O(N^2)`.

The fused paths eliminate these full `N x N` global intermediates.

---

## 4. Main Finding

The result is split:

```text
Memory goal: achieved
Latency goal: not yet achieved
```

The Flash-style implementation has already reduced intermediate global-memory complexity, but the current CUDA execution strategy is inefficient enough that it is still slower than the baseline.

---

## 5. Performance Interpretation

### Naive Fused

Approximate slowdown relative to baseline:

```text
128  -> 1.80× slower
256  -> 1.63× slower
512  -> 1.81× slower
1024 -> 1.92× slower
2048 -> 1.91× slower
```

So kernel fusion alone does not guarantee a speedup.

### Tiled Flash

Approximate slowdown relative to baseline:

```text
128  -> 2.09× slower
256  -> 1.89× slower
512  -> 2.46× slower
1024 -> 2.26× slower
2048 -> 2.19× slower
```

The tiled path is also slower than the naive fused path at every tested sequence length.

---

## 6. Why Tiled Flash Is Currently Slower

The tiled implementation adds repeated per-tile work:

```text
load K/V tile
synchronize
compute QK scores
synchronize
reduce tile max
synchronize
update online max
synchronize
compute exp values
synchronize
reduce tile sum
synchronize
update online sum
synchronize
accumulate output
synchronize
```

This introduces substantial `__syncthreads()` overhead.

More importantly, the current QK mapping is inefficient:

```text
one thread
-> one key
-> loops serially over all 64 head dimensions
```

With `TILE_K = 32`, only the first 32 threads participate in score calculation, while other threads in the 128-thread block are idle during that phase.

Each active score thread performs 64 multiply-adds serially.

That is likely the largest current compute-utilization bottleneck.

---

## 7. Better QK Mapping

The next design should use cooperative dot products:

```text
one warp
-> one Q · K dot product
```

For `head_dim = 64`:

```text
32 lanes
-> each lane handles about 2 dimensions
-> warp reduction combines partial sums
```

Use warp primitives such as:

```cpp
__shfl_down_sync(...)
```

This changes the score calculation from:

```text
1 thread × 64 serial dimensions
```

to roughly:

```text
32 threads × 2 dimensions + warp reduction
```

This should greatly improve utilization.

---

## 8. Synchronization Bottleneck

The current tiled kernel relies heavily on block-wide synchronization.

A better kernel should replace as many shared-memory reductions as possible with warp-level operations.

Priority:

```text
warp-level score reduction
warp-level max reduction
warp-level sum reduction
fewer __syncthreads()
```

This is likely more important than tuning tile size right now.

---

## 9. Memory Savings

The memory improvement is already substantial.

At `seq_len = 2048`:

```text
Baseline temporary global intermediates:
32 MB

Tiled Flash full N×N intermediates:
0 MB
```

The baseline's current logical intermediate traffic reaches:

```text
96 MB
```

at `N = 2048`.

The fused implementations avoid those score/weight global materializations.

This is **logical traffic avoided**, not guaranteed physical DRAM bytes saved, because GPU caches can absorb some accesses.

Actual HBM/DRAM bytes should be confirmed with profiling counters when available.

---

## 10. Why Memory Savings Do Not Yet Produce Lower Latency

Current tradeoff:

```text
Baseline:
more global-memory traffic
+
strong specialized parallel mapping

Tiled Flash:
less global-memory traffic
+
weak QK thread mapping
+
many synchronization barriers
+
idle threads during score computation
```

Right now:

```text
extra compute/synchronization overhead
>
saved memory cost
```

Therefore total latency is worse.

---

## 11. Why the Baseline Is Strong

The baseline kernels each have natural parallel structures:

- **AttentionScores**: 2D grid over `(query, key)`
- **Softmax**: one block per row
- **MatMul**: 2D output-tile mapping

Even though they write/read global intermediates, they expose much more fine-grained GPU parallelism than the current simplified Flash kernel.

---

## 12. Important GPU Optimization Lesson

The experiment proves:

> Fewer kernel launches and fewer global-memory intermediates do not automatically imply lower latency.

A fused kernel can still lose if it has:

```text
lower occupancy
poorer thread utilization
more synchronization
more serialized arithmetic
worse instruction throughput
```

---

## 13. Current Trend

Flash speedup remains around:

```text
0.41× to 0.53×
```

across all tested sequence lengths.

There is no clear sign yet that the current tiled implementation approaches baseline performance as `N` grows.

This suggests the compute/synchronization problem is persistent rather than only a small-input overhead issue.

---

## 14. Recommended Optimization Order

### Priority 1 — Parallelize QK dot product

Replace one-thread-per-dot-product with warp-cooperative dot products.

### Priority 2 — Reduce synchronization

Use warp reductions and warp-level synchronization where possible.

### Priority 3 — Improve K/V load efficiency

Check coalescing and consider vectorized loads (`float4`) after compute mapping improves.

### Priority 4 — Tune `TILE_K` and block size

Only after the kernel mapping is improved, benchmark variants such as:

```text
TILE_K = 16, 32, 64
BLOCK_SIZE = 64, 128, 256
```

---

## 15. Current Phase 5 Status

Completed:

```text
✅ Standard CUDA attention baseline
✅ Naive fused attention
✅ Tiled K/V Flash-style attention
✅ Online softmax
✅ Causal masking
✅ Removal of full N×N global intermediates
✅ Correctness validation
✅ Latency benchmark
✅ Temporary-memory analysis
✅ Logical intermediate-traffic analysis
```

Still needed for the performance objective:

```text
⬜ warp-cooperative QK computation
⬜ fewer synchronization barriers
⬜ improved GPU utilization
⬜ final optimized benchmark
```

---

## 16. Final Conclusion

At `seq_len = 2048`:

```text
Baseline:
0.999916 ms

Naive Fused:
1.910999 ms

Tiled Flash:
2.193050 ms
```

The tiled version is currently about:

```text
2.19× slower than baseline
```

but it simultaneously eliminates:

```text
32 MB
```

of `scores + weights` temporary global memory and avoids approximately:

```text
96 MB
```

of logical intermediate accesses from the current baseline path.

Therefore:

> The algorithmic memory-complexity improvement is working, but the current thread mapping and synchronization strategy prevent the implementation from converting that memory advantage into a latency improvement.

The next optimization should focus first on **parallelizing each QK dot product across a warp**.
