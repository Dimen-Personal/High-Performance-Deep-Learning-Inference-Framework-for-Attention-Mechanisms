# Warp-Optimized Flash Attention Benchmark Analysis

## 1. Benchmark Goal

This experiment evaluates whether the first optimization pass on the tiled Flash Attention kernel successfully improves GPU utilization.

Compared implementations:

1. **Baseline Attention**
   - `AttentionScores`
   - `AttentionSoftmax`
   - `MatMul(weights, V)`

2. **Naive Fused Attention**
   - fuses the three logical attention stages
   - avoids full `N × N` global intermediates
   - stores one score row in shared memory

3. **Original Tiled Flash Attention**
   - K/V tiling
   - online softmax
   - fixed tile-sized shared-memory footprint

4. **Warp-Optimized Tiled Flash Attention**
   - caches Q
   - one warp cooperatively computes one Q·K dot product
   - warp shuffle reductions for QK / tile max / tile sum
   - fewer block-wide synchronizations

Benchmark configuration:

```text
Head Dimension: 64
Warmup Runs: 10
Benchmark Runs: 100
```

---

## 2. Raw Performance Results

| SeqLen | Baseline (ms) | Naive (ms) | Original Tiled (ms) | Optimized Tiled (ms) |
|---:|---:|---:|---:|---:|
| 128 | 0.022098 | 0.039844 | 0.046162 | 0.036782 |
| 256 | 0.047933 | 0.078008 | 0.091105 | 0.070789 |
| 512 | 0.100884 | 0.180675 | 0.245514 | 0.170373 |
| 1024 | 0.300186 | 0.608225 | 0.713318 | 0.427571 |
| 2048 | 0.995850 | 1.891523 | 2.192138 | 1.371023 |

---

## 3. Optimization Gain

| SeqLen | Opt vs Original Tiled | Tiled Latency Reduction | Optimized / Baseline | Still Slower Than Baseline |
|---:|---:|---:|---:|---:|
| 128 | 1.2550× | 20.32% | 0.6008× | 66.45% |
| 256 | 1.2870× | 22.30% | 0.6771× | 47.68% |
| 512 | 1.4410× | 30.61% | 0.5921× | 68.88% |
| 1024 | 1.6683× | 40.06% | 0.7021× | 42.44% |
| 2048 | 1.5989× | 37.46% | 0.7264× | 37.67% |

The first optimization pass is clearly successful.

The optimized tiled kernel is faster than the original tiled implementation at every tested sequence length.

The improvement becomes stronger as sequence length grows:

```text
128  -> 1.255× faster
256  -> 1.287× faster
512  -> 1.441× faster
1024 -> 1.668× faster
2048 -> 1.599× faster
```

This strongly indicates that the original QK thread mapping and shared-memory reductions were real bottlenecks.

---

## 4. Why the Optimization Worked

The original tiled kernel used:

```text
1 thread
-> computes one complete Q · K dot product
-> loops serially through all 64 head dimensions
```

With:

```text
BLOCK_SIZE = 128
TILE_K = 32
head_dim = 64
```

only 32 threads actively computed attention scores, while many other threads were idle during that stage.

The optimized version changed this to:

```text
1 warp
-> computes one Q · K dot product cooperatively
```

For `head_dim = 64`, each lane processes approximately two dimensions, followed by a warp-level reduction.

This improves:

- thread utilization
- instruction-level parallelism
- QK reduction latency
- synchronization overhead

The optimized version also replaces several shared-memory reductions with warp shuffle operations.

That is why performance improves substantially, especially at larger sequence lengths.

---

## 5. Strongest Improvement Region

The largest observed gain is at:

```text
seq_len = 1024
```

Original tiled:

```text
0.713318 ms
```

Optimized tiled:

```text
0.427571 ms
```

Speedup:

```text
1.6683×
```

Latency reduction:

```text
~40.06%
```

At `seq_len = 2048`:

```text
2.192138 ms
-> 1.371023 ms
```

which is approximately:

```text
37.46% lower latency
```

This is a strong result for a single optimization pass.

---

## 6. Comparison With Naive Fused Attention

The optimized tiled implementation also becomes faster than the naive fused implementation for every tested size:

```text
128:
Naive     0.039844 ms
Optimized 0.036782 ms

256:
Naive     0.078008 ms
Optimized 0.070789 ms

512:
Naive     0.180675 ms
Optimized 0.170373 ms

1024:
Naive     0.608225 ms
Optimized 0.427571 ms

2048:
Naive     1.891523 ms
Optimized 1.371023 ms
```

This is important because the previous tiled implementation was slower than the naive fused path.

After the warp optimization, tiling + online softmax finally becomes competitive with, and consistently better than, the simpler fused implementation.

This confirms that the Flash-style tiled structure itself was not the problem; the previous execution mapping was.

---

## 7. Baseline Still Wins

Despite the large improvement, the optimized tiled implementation is still slower than baseline.

At `seq_len = 2048`:

```text
Baseline:
0.995850 ms

Optimized Tiled:
1.371023 ms
```

The optimized tiled kernel is still approximately:

```text
37.67% slower
```

At `seq_len = 1024`:

```text
Baseline:
0.300186 ms

Optimized Tiled:
0.427571 ms
```

which is approximately:

```text
42.44% slower
```

Therefore:

```text
First optimization pass:
successful

Final performance target:
not yet reached
```

---

## 8. Why Baseline Is Still Faster

The remaining performance gap is now less likely to be dominated by the local QK dot-product implementation.

The next major architectural issue is:

```text
1 block = 1 query
```

This means every query block independently reloads the same K/V tiles.

Conceptually:

```text
Query 0 block:
load K tile 0
load V tile 0

Query 1 block:
load K tile 0 again
load V tile 0 again

Query 2 block:
load K tile 0 again
load V tile 0 again
```

So although the optimized kernel no longer materializes `N × N` score and weight matrices, it still has substantial repeated K/V traffic.

A more advanced Flash Attention mapping should process multiple query rows in the same CTA/block:

```text
one block
=
multiple Q rows
×
one K/V tile
```

Then one K/V tile load can be reused across several queries.

That is the next major performance opportunity.

---

## 9. Memory Benefit Remains Unchanged

The optimization changes compute mapping but does not change the memory-complexity advantage.

Baseline temporary global memory:

| SeqLen | Temporary Memory |
|---:|---:|
| 128 | 0.125 MB |
| 256 | 0.500 MB |
| 512 | 2.000 MB |
| 1024 | 8.000 MB |
| 2048 | 32.000 MB |

The tiled Flash implementations avoid these full `scores[N,N]` and `weights[N,N]` intermediates.

At `seq_len = 2048`, the baseline also performs approximately:

```text
96 MB
```

of logical intermediate score/weight accesses according to the current kernel structure.

The Flash paths avoid this global materialization.

This remains a key architectural advantage even though latency has not yet beaten baseline.

---

## 10. Interpretation of the Scaling Trend

The optimization gain increases with larger sequence length:

```text
1.255×
1.287×
1.441×
1.668×
1.599×
```

This is a good sign.

It suggests that warp-based QK computation becomes increasingly valuable as the number of tiles and attention computations grows.

The slight reduction from `1.668×` at 1024 to `1.599×` at 2048 does not invalidate the trend.

Possible causes include:

- occupancy limits
- shared-memory pressure
- cache behavior
- more repeated K/V loads
- output accumulation cost
- memory subsystem pressure

The next architecture change should specifically target K/V reuse.

---

## 11. Recommended Next Optimization

The next step should be:

# Multi-query-per-block tiling

Instead of:

```text
1 block = 1 query
```

move toward:

```text
1 block = Br query rows
```

For example:

```text
Br = 4
Br = 8
or
Br = 16
```

Then:

```text
load one K/V tile
```

and reuse it for all query rows inside the block.

Conceptually:

```text
Q0 ┐
Q1 │
Q2 │
Q3 ├── use the same K/V tile
...│
QBr┘
```

This should reduce repeated global-memory traffic and move the implementation closer to the IO-aware design of real Flash Attention.

---

## 12. Recommended Optimization Order From Here

1. **Multi-query-per-block / Q tiling**
   - highest priority

2. **Reuse K/V tile across multiple Q rows**
   - core IO-aware optimization

3. **Profile occupancy and shared-memory pressure**
   - especially after increasing queries per CTA

4. **Tune Br / TILE_K / BLOCK_SIZE**
   - after the new architecture works

5. **Vectorized global loads**
   - e.g. `float4`

6. **Further warp specialization**
   - separate load / compute responsibilities if useful

7. **FP16 / Tensor Core path**
   - later optimization stage

---

## 13. Phase 5 Progress

Completed:

```text
✅ Baseline CUDA Attention
✅ Naive fused attention
✅ Tiled K/V attention
✅ Online softmax
✅ Causal masking
✅ N×N intermediate elimination
✅ Correctness validation
✅ Performance benchmark
✅ Memory analysis
✅ Warp-cooperative QK
✅ Warp max reduction
✅ Warp sum reduction
✅ Reduced synchronization
```

Current result:

```text
Original Tiled:
2.192138 ms at N=2048

Optimized Tiled:
1.371023 ms at N=2048
```

This is:

```text
1.5989× faster
```

than the original tiled kernel.

Still remaining:

```text
⬜ Multiple query rows per CTA
⬜ K/V reuse across Q rows
⬜ Further shared-memory / occupancy tuning
⬜ Final attempt to beat baseline latency
```

---

## 14. Final Conclusion

The first optimization pass clearly succeeded.

The data confirms that:

> The previous one-thread-per-QK mapping and block-wide reduction strategy were major bottlenecks.

Warp-cooperative score computation and warp-level reductions improve tiled Flash Attention by approximately:

```text
1.25× to 1.67×
```

across the tested sequence lengths.

The optimized kernel is now consistently faster than both the original tiled implementation and the naive fused implementation.

However, it is still slower than the specialized baseline.

The next major bottleneck is architectural rather than local:

> each block still processes only one query, causing repeated loading of the same K/V tiles.

Therefore the next optimization should move to:

```text
multiple queries per block
+
shared K/V tile reuse
```

which is the next step toward a more faithful high-performance Flash Attention implementation.
