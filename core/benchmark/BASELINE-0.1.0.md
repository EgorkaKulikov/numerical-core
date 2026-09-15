# Performance baseline of numerical-core 0.1.0

Microbenchmark of the public API (`LinearAlgebra.solve/matMat/matVec/matTransVec/atWa`,
`Conditioning.conditionInf/symmetricEigenvalues`) on version 0.1.0 before the rework; the inputs are deterministic
(`java.util.Random(1000 + n)`, diagonally dominant matrices). Machine: Apple M1 Pro, 8 cores, macOS, JDK 21,
default backend (multik 0.2.3 / OpenBLAS). To reproduce: `./gradlew :numerical-core:benchmark -Pbench.args="256 512 1024 2048"`
(source: `src/benchmark/kotlin/numerics/bench/Bench.kt`, source set `benchmark`, not part of the artifact).

## Result

```
backend: multik-cpu (OpenBLAS)
processors: 8
jvm: 21.0.10
os.arch: aarch64
date: 2026-09-08T23:49:45.432893+03:00

Time in ms: median of 5 measurements after 3 warm-ups (conditionInf and symmetricEigenvalues — median of 3).
conditionInf is measured only for n ≤ 512, symmetricEigenvalues only for n ≤ 256; otherwise "—" (too slow).

| n | solve | matMat | matVec | matTransVec | atWa | conditionInf | symmetricEigenvalues |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2.05 | 11.73 | 4.13 | 5.54 | 12.24 | 294.35 | 90.21 |
| 512 | 4.40 | 31.16 | 4.04 | 5.84 | 20.86 | 2196.11 | — |
| 1024 | 37.66 | 37.26 | 3.72 | 7.43 | 42.68 | — | — |
| 2048 | 179.82 | 210.71 | 7.83 | 35.34 | 197.94 | — | — |

sink: double[][]
```
