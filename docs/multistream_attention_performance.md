# Multi-Stream End-to-End Attention Performance Log

## Overview

This benchmark evaluates the performance impact of using multiple CUDA streams for concurrent end-to-end Attention inference requests.

The benchmark compares:

- **Serial Execution**
  - Request A completes entirely before Request B starts.
- **Multi-Stream Execution**
  - Request A and Request B are submitted to two independent CUDA streams before synchronization.

Each Attention request includes:

```text
Pinned Host Q / K / V
        ↓
Async H2D Transfer
        ↓
AttentionScores
        ↓
AttentionSoftmax
        ↓
MatMul(weights, V)
        ↓
Async D2H Transfer
        ↓
Pinned Host Output
```

The multi-stream benchmark therefore measures **request-level concurrency and throughput improvement**, rather than accelerating the internal dependency chain of a single Attention request.

---

## Benchmark Configuration

### GPU Execution Model

Each Attention pipeline owns:

- One independent `Executor`
- One independent CUDA stream
- One independent CUDA memory allocation set
- Pinned host input/output buffers
- Asynchronous H2D / D2H transfers

The two concurrent pipelines are executed as:

```text
Stream A:
H2D(QA, KA, VA)
→ AttentionScores A
→ AttentionSoftmax A
→ MatMul A
→ D2H Output A

Stream B:
H2D(QB, KB, VB)
→ AttentionScores B
→ AttentionSoftmax B
→ MatMul B
→ D2H Output B
```

Synchronization occurs only after both pipelines have been submitted.

---

## Parameters

```text
Head Dimension: 64
Warmup Runs: 3
Benchmark Runs: 10

Sequence Lengths:
128
256
512
1024
2048
```

---

## Results

| SeqLen | HeadDim | Serial (ms) | Multi-Stream (ms) | Speedup | Serial req/s | Multi req/s |
|------:|--------:|------------:|------------------:|--------:|-------------:|------------:|
| 128  | 64 | 0.1141 | 0.0960 | 1.1885x | 17526.7480 | 20831.3804 |
| 256  | 64 | 0.1910 | 0.1348 | 1.4166x | 10471.8457 | 14834.3852 |
| 512  | 64 | 0.3316 | 0.2557 | 1.2968x | 6030.4992 | 7820.2622 |
| 1024 | 64 | 0.8873 | 0.7298 | 1.2157x | 2254.0024 | 2740.2959 |
| 2048 | 64 | 2.8045 | 2.4089 | 1.1642x | 713.1488 | 830.2437 |

---

## Performance Improvement

| SeqLen | Speedup | Throughput Improvement |
|------:|--------:|-----------------------:|
| 128  | 1.1885x | +18.85% |
| 256  | 1.4166x | +41.66% |
| 512  | 1.2968x | +29.68% |
| 1024 | 1.2157x | +21.57% |
| 2048 | 1.1642x | +16.42% |

The best observed result was:

```text
Sequence Length: 256
Speedup: 1.4166x
Serial Throughput: 10471.85 requests/s
Multi-Stream Throughput: 14834.39 requests/s
```

This corresponds to approximately:

```text
+41.66% throughput improvement
```

---

## Latency Reduction

Speedup and latency reduction are not the same metric.

Latency reduction can be calculated as:

```text
Latency Reduction
=
1 - MultiStreamTime / SerialTime
```

### Example: SeqLen = 256

```text
Serial Time = 0.1910 ms
Multi-Stream Time = 0.1348 ms

Latency Reduction
=
1 - 0.1348 / 0.1910
≈ 29.4%
```

Therefore:

```text
Speedup:          1.4166x
Throughput Gain:  +41.66%
Total Time Drop:  ~29.4%
```

---

## Observed Trend

The measured speedup follows approximately:

```text
SeqLen 128  → 1.1885x
SeqLen 256  → 1.4166x
SeqLen 512  → 1.2968x
SeqLen 1024 → 1.2157x
SeqLen 2048 → 1.1642x
```

The performance improvement first increases and then gradually decreases as sequence length grows.

The best performance occurs around:

```text
seq_len = 256
```

---

## Interpretation

### Small Sequence Lengths

At very small sequence lengths, fixed overheads such as:

- CUDA kernel launch overhead
- stream scheduling
- H2D / D2H setup
- Executor scheduling

represent a larger portion of total execution time.

As a result, the available concurrency is not fully utilized.

For example:

```text
SeqLen = 128
Speedup = 1.1885x
```

---

### Medium Sequence Lengths

At medium sequence lengths, each request contains enough work for the GPU to overlap operations from independent CUDA streams.

Possible overlap includes:

```text
Request A compute
        +
Request B H2D transfer
```

or:

```text
Request A kernel execution
        +
Request B kernel execution
```

or:

```text
Request A D2H
        +
Request B compute
```

This is likely why the highest speedup appears at:

```text
SeqLen = 256
Speedup = 1.4166x
```

---

### Large Sequence Lengths

As sequence length increases, Attention becomes significantly more expensive.

The Attention score matrix scales approximately as:

```text
O(seq_len²)
```

For example:

```text
1024² = 1,048,576

2048² = 4,194,304
```

Doubling sequence length therefore increases the score matrix size by approximately 4x.

At larger sizes, a single Attention request consumes more:

- SM compute resources
- memory bandwidth
- shared memory / cache resources
- memory traffic

This leaves fewer idle GPU resources available for a second stream.

The measured trend supports this:

```text
SeqLen 256  → 1.4166x
SeqLen 512  → 1.2968x
SeqLen 1024 → 1.2157x
SeqLen 2048 → 1.1642x
```

This suggests that multi-stream concurrency becomes less beneficial as the GPU approaches higher resource utilization.

---

## What the Multi-Stream Optimization Is Actually Improving

The current implementation does **not** parallelize the internal dependency chain of one Attention operation.

A single request still executes:

```text
AttentionScores
      ↓
AttentionSoftmax
      ↓
MatMul
```

These operations depend on one another and therefore remain ordered.

Instead, multi-stream execution improves:

```text
Multiple independent Attention requests
```

For example:

```text
Request A
and
Request B
```

are allowed to execute concurrently on independent streams.

Therefore, the primary optimization target is:

```text
Request-level throughput
```

rather than:

```text
Single-request latency
```

---

## CUDA Features Used

The optimized execution path currently includes:

```text
Executor-owned CUDA streams
        ↓
cudaMemcpyAsync
        ↓
Pinned host memory
        ↓
Independent Executors
        ↓
Independent CUDA streams
        ↓
Concurrent Attention requests
```

The corresponding execution model is:

```text
CPU
 │
 ├── enqueue Request A
 │
 ├── enqueue Request B
 │
 └── synchronize
      ↓
GPU
 ├── Stream A
 │    H2D → Attention → D2H
 │
 └── Stream B
      H2D → Attention → D2H
```

---

## Current Conclusion

Multi-stream execution provides a consistent performance improvement across all tested sequence lengths.

Observed speedup range:

```text
1.1642x - 1.4166x
```

Maximum measured improvement:

```text
1.4166x
```

at:

```text
Sequence Length = 256
Head Dimension = 64
```

The results demonstrate that request-level CUDA stream concurrency can significantly improve inference throughput when sufficient GPU resources are available for overlapping independent workloads.

The declining speedup at larger sequence lengths suggests increasing GPU resource saturation.

---

## Next Profiling Step

The next step is to validate the cause of the performance improvement using NVIDIA Nsight Systems.

Recommended profiling cases:

```text
SeqLen = 256
```

Reason:

```text
Highest measured speedup: 1.4166x
```

and:

```text
SeqLen = 2048
```

Reason:

```text
Lower measured speedup: 1.1642x
```

These two cases provide a useful comparison between:

```text
High concurrency benefit
vs.
Possible GPU resource saturation
```

Nsight Systems should be used to inspect:

- H2D transfer overlap
- D2H transfer overlap
- kernel concurrency
- stream overlap
- GPU idle gaps
- copy engine utilization
- relative kernel execution timing

A later Nsight Compute analysis can further investigate:

- SM utilization
- memory bandwidth utilization
- occupancy
- warp stalls
- memory bottlenecks

---

## Summary

```text
Optimization:
Pinned Memory
+
cudaMemcpyAsync
+
Multiple CUDA Streams
+
Independent Attention Requests

Best Speedup:
1.4166x

Best Sequence Length:
256

Best Throughput:
14834.39 requests/s

Main Benefit:
Higher concurrent inference throughput

Main Limitation:
Reduced overlap benefit as Attention workload becomes large
and GPU resources approach saturation
```
