# INT8 MatMul OpenMP Scaling Report

## 1. Objective

This experiment evaluates the multi-thread scaling behavior of the INT8 MatMul implementation using OpenMP.

The INT8 compute path is:

```text
INT8 A × INT8 B
→ INT32 accumulation
→ FP32 output
```

Quantization is performed before timing, so the benchmark isolates INT8 compute scaling.

Tested worker counts:

```text
1
2
4
8
```

Tested matrix sizes:

```text
256 × 256
512 × 512
1024 × 1024
```

## 2. Experimental Setup

```text
Warmup Runs: 3
Benchmark Runs: 10
OpenMP Max Threads: 10
OpenMP Dynamic Threads: 0
```

The implementation uses row-level static scheduling:

```cpp
#pragma omp parallel for schedule(static)
```

Each worker writes to a separate subset of output rows, so no mutex, atomic operation, or critical section is required in the main MatMul loop.

## 3. Metrics

```text
Speedup(p) = T1 / Tp
Efficiency(p) = Speedup(p) / p
```

Ideal scaling:

```text
2 workers → 2.0× → 100%
4 workers → 4.0× → 100%
8 workers → 8.0× → 100%
```

## 4. Results: 256 × 256

| Workers | Time (ms) | Speedup | Efficiency | MAE | Correct |
|---:|---:|---:|---:|---:|---|
| 1 | 71.5449 | 1.0000× | 100.00% | 0.023688 | PASS |
| 2 | 35.9148 | 1.9921× | 99.60% | 0.023688 | PASS |
| 4 | 18.1122 | 3.9501× | 98.75% | 0.023688 | PASS |
| 8 | 14.4609 | 4.9475× | 61.84% | 0.023688 | PASS |

Scaling is almost ideal up to 4 workers. At 8 workers, efficiency drops sharply to 61.84%. The most likely cause is that the workload becomes too small per worker, so OpenMP scheduling, worker activation, synchronization, and memory/cache overhead become significant relative to useful computation.

The transition from 4 to 8 workers reduces latency from 18.1122 ms to 14.4609 ms, only about a 20.2% additional latency reduction despite doubling the worker count.

## 5. Results: 512 × 512

| Workers | Time (ms) | Speedup | Efficiency | MAE | Correct |
|---:|---:|---:|---:|---:|---|
| 1 | 575.5692 | 1.0000× | 100.00% | 0.033505 | PASS |
| 2 | 289.8251 | 1.9859× | 99.30% | 0.033505 | PASS |
| 4 | 146.1051 | 3.9394× | 98.49% | 0.033505 | PASS |
| 8 | 74.9840 | 7.6759× | 95.95% | 0.033505 | PASS |

This workload scales extremely well. The 8-worker version reduces latency from 575.5692 ms to 74.9840 ms, approximately an 86.97% reduction.

The 95.95% efficiency at 8 workers indicates that this workload is large enough to amortize OpenMP synchronization and scheduling overhead.

## 6. Results: 1024 × 1024

| Workers | Time (ms) | Speedup | Efficiency | MAE | Correct |
|---:|---:|---:|---:|---:|---|
| 1 | 7015.9952 | 1.0000× | 100.00% | 0.047411 | PASS |
| 2 | 3583.0195 | 1.9581× | 97.91% | 0.047411 | PASS |
| 4 | 1825.1998 | 3.8440× | 96.10% | 0.047411 | PASS |
| 8 | 917.0603 | 7.6505× | 95.63% | 0.047411 | PASS |

The 1024 workload also scales very strongly. The 8-worker version reduces latency from 7015.9952 ms to 917.0603 ms, approximately an 86.93% reduction.

## 7. Cross-Workload Comparison

### Speedup

| Matrix Size | 2 Workers | 4 Workers | 8 Workers |
|---:|---:|---:|---:|
| 256 | 1.9921× | 3.9501× | 4.9475× |
| 512 | 1.9859× | 3.9394× | 7.6759× |
| 1024 | 1.9581× | 3.8440× | 7.6505× |

### Efficiency

| Matrix Size | 2 Workers | 4 Workers | 8 Workers |
|---:|---:|---:|---:|
| 256 | 99.60% | 98.75% | 61.84% |
| 512 | 99.30% | 98.49% | 95.95% |
| 1024 | 97.91% | 96.10% | 95.63% |

The key trend is:

```text
256:
excellent scaling up to 4 workers
poor scaling at 8 workers

512:
excellent scaling through 8 workers

1024:
excellent scaling through 8 workers
```

## 8. Workload Size and Parallel Scaling

The experiment shows that parallel efficiency strongly depends on workload size.

For 256 × 256, the useful computation assigned to each worker becomes too small at 8 workers, so fixed parallel overhead becomes significant.

For 512 × 512 and 1024 × 1024, each worker receives enough computation that runtime overhead is effectively amortized.

This demonstrates an important principle:

```text
More threads are not automatically better.
```

The optimal worker count depends on the amount of work available per thread.

## 9. Synchronization Overhead

The OpenMP loop has an implicit barrier at the end of the parallel region.

The 256 × 256 result clearly demonstrates synchronization/runtime overhead:

```text
4 workers:
98.75% efficiency

8 workers:
61.84% efficiency
```

For 512 and 1024, however:

```text
8-worker efficiency ≈ 96%
```

Therefore the synchronization cost itself is not inherently large. It becomes large relative to useful work only when the workload per worker is too small.

## 10. Correctness

MAE remains unchanged across thread counts for each matrix size:

```text
256:
0.023688

512:
0.033505

1024:
0.047411
```

All configurations pass correctness validation.

This confirms that the OpenMP row-level implementation introduces no race conditions and that multi-threading changes performance without changing numerical behavior.

## 11. Comparison with Earlier FP32 Scaling

Earlier FP32 results at 8 workers were approximately:

```text
256:
7.4245×, 92.81%

512:
7.5219×, 94.02%

1024:
7.4719×, 93.40%
```

Current INT8 results:

```text
256:
4.9475×, 61.84%

512:
7.6759×, 95.95%

1024:
7.6505×, 95.63%
```

For medium and large workloads, INT8 scales at least as well as FP32.

At 256, INT8 reaches a clear saturation point beyond 4 workers.

## 12. Practical Thread Selection

Based on the current results:

```text
256 × 256:
4 workers is the most efficient choice

512 × 512:
8 workers is effective

1024 × 1024:
8 workers is effective
```

At 256, 8 workers still minimize absolute latency, but 4 workers are much more efficient in terms of CPU resource usage.

Therefore thread selection should depend on whether the objective is:

```text
minimum latency
```

or:

```text
maximum CPU efficiency / better concurrent throughput
```

## 13. Main Findings

1. INT8 MatMul now supports OpenMP multi-thread execution.
2. Numerical correctness is preserved across all worker counts.
3. Two-worker scaling is nearly ideal for all tested workloads.
4. Four-worker scaling is also extremely strong.
5. 512 and 1024 achieve approximately 7.65× speedup with 8 workers.
6. 8-worker efficiency remains around 96% for 512 and 1024.
7. The 256 workload saturates after 4 workers.
8. At 256, 8-worker efficiency drops to 61.84%.
9. This directly demonstrates synchronization/runtime overhead when per-worker workload becomes too small.
10. INT8 multi-core scaling is highly effective for medium and large workloads.
11. Optimal worker count should depend on workload size.
12. The INT8 multi-threading requirement of Phase 4 is effectively complete at the operator level.

## 14. Final Conclusion

The OpenMP INT8 MatMul implementation demonstrates excellent multi-core scaling for medium and large workloads.

Strongest results:

```text
512 × 512:
1 worker = 575.5692 ms
8 workers = 74.9840 ms
speedup = 7.6759×
efficiency = 95.95%

1024 × 1024:
1 worker = 7015.9952 ms
8 workers = 917.0603 ms
speedup = 7.6505×
efficiency = 95.63%
```

The 256 × 256 workload provides the contrasting case:

```text
4 workers:
98.75% efficiency

8 workers:
61.84% efficiency
```

This experiment therefore demonstrates both effective multi-core scaling and the limits of over-parallelization, directly addressing the Phase 4 requirement to analyze scaling efficiency and thread synchronization overhead.
