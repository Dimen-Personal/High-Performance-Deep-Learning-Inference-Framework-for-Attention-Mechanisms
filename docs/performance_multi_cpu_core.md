# CPU MatMul OpenMP Scaling Report

## 1. Objective

This experiment evaluates how well the CPU FP32 MatMul implementation scales as the number of OpenMP workers increases.

The goal is to study:

- latency reduction,
- speedup,
- parallel efficiency,
- workload-size effects,
- and synchronization / scheduling overhead.

The same MatMul algorithm is used for all runs.

Only the OpenMP worker count changes.

---

## 2. Experimental Setup

The benchmark was executed on the cloud server.

Tested matrix sizes:

```text
256 × 256
512 × 512
1024 × 1024
```

For each workload:

```text
Warmup Runs: 3
Benchmark Runs: 10
OpenMP Dynamic Threads: Disabled
```

Worker counts:

```text
1
2
4
8
```

All tested configurations passed correctness validation.

The OpenMP implementation uses static row-level parallelism:

```cpp
#pragma omp parallel for schedule(static)
```

Each worker processes a different subset of output rows.

---

## 3. Metrics

Speedup is defined as:

```text
Speedup(p) = T1 / Tp
```

where:

```text
T1 = execution time with one worker
Tp = execution time with p workers
```

Parallel efficiency is:

```text
Efficiency(p) = Speedup(p) / p
```

Ideal scaling would therefore be:

```text
2 workers → 2.0× speedup → 100% efficiency
4 workers → 4.0× speedup → 100% efficiency
8 workers → 8.0× speedup → 100% efficiency
```

---

# 4. Results: 256 × 256

| Workers | Time (ms) | Speedup | Efficiency | Correct |
|---:|---:|---:|---:|---|
| 1 | 47.0130 | 1.0000× | 100.00% | PASS |
| 2 | 23.7126 | 1.9826× | 99.13% | PASS |
| 4 | 12.4596 | 3.7732× | 94.33% | PASS |
| 8 | 6.3321 | 7.4245× | 92.81% | PASS |

### Analysis

The two-worker result is nearly ideal:

```text
1.9826× speedup
99.13% efficiency
```

At four workers:

```text
3.7732× speedup
94.33% efficiency
```

At eight workers:

```text
7.4245× speedup
92.81% efficiency
```

Execution time decreases from:

```text
47.0130 ms
```

to:

```text
6.3321 ms
```

which is approximately:

```text
1 - 6.3321 / 47.0130
≈ 86.53%
```

lower latency.

Even at the relatively small 256 × 256 workload, OpenMP overhead remains small enough that scaling stays above 92% efficiency.

---

# 5. Results: 512 × 512

| Workers | Time (ms) | Speedup | Efficiency | Correct |
|---:|---:|---:|---:|---|
| 1 | 663.2321 | 1.0000× | 100.00% | PASS |
| 2 | 336.2895 | 1.9722× | 98.61% | PASS |
| 4 | 178.4945 | 3.7157× | 92.89% | PASS |
| 8 | 88.1729 | 7.5219× | 94.02% | PASS |

### Analysis

Two workers achieve:

```text
1.9722× speedup
98.61% efficiency
```

Four workers achieve:

```text
3.7157× speedup
92.89% efficiency
```

Eight workers achieve:

```text
7.5219× speedup
94.02% efficiency
```

Execution time decreases from:

```text
663.2321 ms
```

to:

```text
88.1729 ms
```

which corresponds to approximately:

```text
1 - 88.1729 / 663.2321
≈ 86.70%
```

lower latency.

The slight increase in efficiency from four to eight workers is not problematic. Real benchmark measurements are influenced by cache state, CPU frequency, OS scheduling, and normal timing noise.

---

# 6. Results: 1024 × 1024

| Workers | Time (ms) | Speedup | Efficiency | Correct |
|---:|---:|---:|---:|---|
| 1 | 6455.9648 | 1.0000× | 100.00% | PASS |
| 2 | 3269.9484 | 1.9743× | 98.72% | PASS |
| 4 | 1672.1210 | 3.8609× | 96.52% | PASS |
| 8 | 864.0343 | 7.4719× | 93.40% | PASS |

### Analysis

Two workers achieve:

```text
1.9743× speedup
98.72% efficiency
```

Four workers achieve:

```text
3.8609× speedup
96.52% efficiency
```

Eight workers achieve:

```text
7.4719× speedup
93.40% efficiency
```

Execution time decreases from:

```text
6455.9648 ms
```

to:

```text
864.0343 ms
```

which corresponds to approximately:

```text
1 - 864.0343 / 6455.9648
≈ 86.62%
```

lower latency.

The four-worker efficiency is especially strong at this workload size, reaching:

```text
96.52%
```

This indicates that the larger matrix provides enough computation per worker to amortize OpenMP overhead very effectively.

---

# 7. Cross-Workload Comparison

## 7.1 Speedup Comparison

| Matrix Size | 2 Workers | 4 Workers | 8 Workers |
|---:|---:|---:|---:|
| 256 | 1.9826× | 3.7732× | 7.4245× |
| 512 | 1.9722× | 3.7157× | 7.5219× |
| 1024 | 1.9743× | 3.8609× | 7.4719× |

The scaling behavior is remarkably consistent across all three workloads.

At two workers:

```text
~1.97–1.98×
```

At four workers:

```text
~3.72–3.86×
```

At eight workers:

```text
~7.42–7.52×
```

There is no major collapse in scaling as matrix size increases.

---

## 7.2 Efficiency Comparison

| Matrix Size | 2 Workers | 4 Workers | 8 Workers |
|---:|---:|---:|---:|
| 256 | 99.13% | 94.33% | 92.81% |
| 512 | 98.61% | 92.89% | 94.02% |
| 1024 | 98.72% | 96.52% | 93.40% |

All multi-worker configurations remain above approximately:

```text
92%
```

parallel efficiency.

This is a strong result.

---

# 8. Workload-Size Effect

The initial expectation was that larger workloads might show better scaling because OpenMP scheduling and synchronization overhead would represent a smaller fraction of total runtime.

The measured results partially support this, but the effect is not dramatic because scaling is already very strong at 256 × 256.

For example, four-worker efficiency is:

```text
256  → 94.33%
512  → 92.89%
1024 → 96.52%
```

The 1024 workload does show the strongest four-worker efficiency, but the 512 result is slightly lower than 256.

This demonstrates that benchmark scaling is influenced by more than just workload size.

Possible factors include:

```text
cache behavior
memory traffic
CPU frequency
OS scheduling
runtime noise
```

Therefore, the correct conclusion is not:

```text
larger matrices always scale better
```

but rather:

```text
all tested matrix sizes scale very well,
with only modest variation in efficiency.
```

---

# 9. Why the Scaling Is Strong

The MatMul implementation parallelizes the outer row loop:

```cpp
#pragma omp parallel for schedule(static)

for(int i = 0; i < M; i++)
{
    ...
}
```

Each worker writes to a separate set of rows in matrix C.

This provides several advantages.

### 9.1 No Write Conflicts

Different workers write to different output locations.

Therefore the main computation does not require:

```text
mutexes
atomic operations
critical sections
```

This avoids expensive synchronization inside the MatMul loop.

### 9.2 Regular Work Distribution

Every output row performs approximately the same amount of computation.

Therefore:

```text
schedule(static)
```

provides balanced work distribution with very low scheduling overhead.

### 9.3 High Compute-to-Synchronization Ratio

MatMul performs a large amount of arithmetic before reaching the final OpenMP barrier.

Therefore synchronization overhead is small relative to useful computation.

---

# 10. Synchronization Overhead

OpenMP introduces several sources of overhead:

```text
worker startup / activation
loop partitioning
runtime scheduling
final barrier synchronization
```

If these costs were significant relative to the workload, parallel efficiency would fall rapidly as the number of workers increased.

That is not observed here.

At eight workers:

```text
256  → 92.81%
512  → 94.02%
1024 → 93.40%
```

All three workloads preserve more than 92% efficiency.

This strongly suggests that synchronization and scheduling overhead are small compared with the MatMul compute cost.

---

# 11. Scaling Loss Relative to Ideal

At eight workers, ideal speedup is:

```text
8.0×
```

Measured speedups are:

```text
256  → 7.4245×
512  → 7.5219×
1024 → 7.4719×
```

The lost scaling is therefore relatively small.

Approximate efficiency loss:

```text
256:
100% - 92.81% = 7.19%

512:
100% - 94.02% = 5.98%

1024:
100% - 93.40% = 6.60%
```

This lost efficiency represents the combined effect of:

```text
OpenMP runtime overhead
barrier synchronization
memory-system contention
cache effects
OS scheduling
possible CPU frequency behavior
```

The benchmark does not isolate these effects individually, but the total penalty is small.

---

# 12. Correctness

Every tested configuration passes output validation.

```text
256:
1 / 2 / 4 / 8 workers → PASS

512:
1 / 2 / 4 / 8 workers → PASS

1024:
1 / 2 / 4 / 8 workers → PASS
```

This confirms that row-level parallelization introduces no race conditions in the current MatMul implementation.

---

# 13. Important Experimental Limitation

These measurements were collected on the cloud server.

The server reports:

```text
OpenMP Max Threads: 10
```

but the physical CPU topology has not yet been verified.

Therefore the technically precise description of these results is:

```text
1 / 2 / 4 / 8 OpenMP worker scaling
```

rather than guaranteed:

```text
1 / 2 / 4 / 8 physical-core scaling
```

To verify whether each worker can correspond to a distinct physical core, inspect:

```bash
lscpu

lscpu -e=CPU,CORE,SOCKET,NODE
```

If the topology confirms at least eight independent physical cores and the runtime is configured accordingly, the experiment can then be described more strictly as physical-core scaling.

---

# 14. Summary Table

| Matrix | 1 Worker | 2 Workers | 4 Workers | 8 Workers | 8-Worker Speedup | 8-Worker Efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 47.0130 ms | 23.7126 ms | 12.4596 ms | 6.3321 ms | 7.4245× | 92.81% |
| 512 | 663.2321 ms | 336.2895 ms | 178.4945 ms | 88.1729 ms | 7.5219× | 94.02% |
| 1024 | 6455.9648 ms | 3269.9484 ms | 1672.1210 ms | 864.0343 ms | 7.4719× | 93.40% |

---

# 15. Main Findings

```text
1. OpenMP MatMul scaling is close to linear across all tested workloads.

2. Two workers consistently achieve approximately 1.97–1.98× speedup.

3. Four workers achieve approximately 3.72–3.86× speedup.

4. Eight workers achieve approximately 7.42–7.52× speedup.

5. Parallel efficiency remains above approximately 92% for every tested multi-worker configuration.

6. Eight workers reduce execution time by approximately 86.5–86.7% compared with one worker.

7. Scaling remains strong from 256×256 through 1024×1024.

8. Static row partitioning provides effective load balancing.

9. Synchronization and scheduling overhead are small relative to MatMul computation.

10. The workload-size effect exists, but it is modest because all tested workloads already have enough computation to amortize OpenMP overhead.

11. All tested configurations preserve numerical correctness.

12. Server CPU topology should be verified before calling the results strict physical-core scaling.
```

---

# 16. Final Conclusion

The CPU FP32 MatMul implementation demonstrates excellent OpenMP strong scaling.

Across matrix sizes:

```text
256 × 256
512 × 512
1024 × 1024
```

the eight-worker configuration consistently achieves approximately:

```text
7.5× speedup
```

with approximately:

```text
93% parallel efficiency
```

The largest measured eight-worker speedup is:

```text
512 × 512
→ 7.5219×
```

The strongest four-worker efficiency is:

```text
1024 × 1024
→ 96.52%
```

The data shows that the current MatMul implementation is highly parallelizable and that OpenMP synchronization overhead is small relative to computation for all tested workload sizes.

This completes the first CPU multi-thread scaling analysis for the FP32 MatMul operator.
