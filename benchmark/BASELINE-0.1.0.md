# Базовая линия производительности numerical-core 0.1.0

Микробенчмарк публичного API (`LinearAlgebra.solve/matMat/matVec/matTransVec/atWa`,
`Conditioning.conditionInf/symmetricEigenvalues`) на версии 0.1.0 до переработки; входы детерминированы
(`java.util.Random(1000 + n)`, диагонально доминирующие матрицы). Машина: Apple M1 Pro, 8 ядер, macOS, JDK 21,
бэкенд по умолчанию (multik 0.2.3 / OpenBLAS). Повторить: `./gradlew benchmark -Pbench.args="256 512 1024 2048"`
(исходник — `src/benchmark/kotlin/numerics/bench/Bench.kt`, source-set `benchmark`, в артефакт не входит).

## Результат

```
backend: multik-cpu (OpenBLAS)
processors: 8
jvm: 21.0.10
os.arch: aarch64
date: 2026-09-08T23:49:45.432893+03:00

Время в мс: медиана из 5 замеров после 3 прогревов (conditionInf и symmetricEigenvalues — медиана из 3).
conditionInf измеряется только при n ≤ 512, symmetricEigenvalues — при n ≤ 256; иначе «—» (слишком долго).

| n | solve | matMat | matVec | matTransVec | atWa | conditionInf | symmetricEigenvalues |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 2.05 | 11.73 | 4.13 | 5.54 | 12.24 | 294.35 | 90.21 |
| 512 | 4.40 | 31.16 | 4.04 | 5.84 | 20.86 | 2196.11 | — |
| 1024 | 37.66 | 37.26 | 3.72 | 7.43 | 42.68 | — | — |
| 2048 | 179.82 | 210.71 | 7.83 | 35.34 | 197.94 | — | — |

sink: double[][]
```
