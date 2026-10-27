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
