# Performance

## Target regime

The library is designed for dense matrices of size `10²…10⁴`. In this range the running
time is determined by the BLAS/LAPACK implementation: the operations `solve`, `matMat`,
`cholesky`, `symmetricEigenvalues` have complexity `O(n³)`, while `matVec` and
`conditionEstimate` (after the factorization) are `O(n²)`. The library adds no overhead of
its own on top of this and does not limit the parallelism of the system implementation.

## Data path

`DenseMatrix` stores its elements in column-major order (`data: DoubleArray`, element `(i, j)`
is at index `i + j·rows`), i.e. in the order expected by BLAS and LAPACK. The matrix is
passed to the BLAS/LAPACK implementation without reordering or transposition. Multiplications
(`matVec`, `matMat`) and norms operate on the matrix array directly. In operations where
LAPACK overwrites its input (`solve`, LU factorization, Cholesky factorization, spectrum of a
symmetric matrix) one copy of matrix size is made, so the original matrices are never
modified. The result is written directly into a new `DoubleArray`. The constructor
`DenseMatrix.fromColumnMajor` accepts a ready-made array without copying; `fromRows` and
`build` create a new array once.

The overloads of `LinearAlgebra` and `Conditioning` methods that accept `Array<DoubleArray>`
copy the data into column-major order on every call. For repeated computations with the same
matrix, build a `DenseMatrix` once and pass it.

## Multithreading

Inside linear algebra operations the library creates no threads: parallelism is provided by
the BLAS/LAPACK implementation and controlled by its environment variables — `OMP_NUM_THREADS`
and `MKL_NUM_THREADS` for Intel MKL, `OPENBLAS_NUM_THREADS` for OpenBLAS,
`VECLIB_MAXIMUM_THREADS` for Apple Accelerate. On a cluster compute node these variables are
usually set by the batch job scheduler.

The only component in which the library itself distributes work across threads is
`ParallelAssembly`: row-wise matrix assembly runs on the common `ForkJoinPool`, the degree of
parallelism is set by `NumericsContext.parallelism`, and `parallel = false` switches to
sequential execution for debugging and reproducibility.

All public methods are thread-safe: different threads may solve different systems
concurrently through a single linear algebra implementation instance.

## Measurements

Machine: Apple M1 Pro, 8 cores, macOS, JDK 21; system implementation — Apple Accelerate.
Times are given in milliseconds as the median of 5 runs after 3 warm-ups (for
`conditionInf` and `symmetricEigenvalues` — the median of 3). Inputs are deterministic
(`java.util.Random(1000 + n)`, diagonally dominant matrices). Run-to-run variation reaches
±30 %, so orders of magnitude and ratios are meaningful, not the second digit. The full
table for version 0.1.0 is in `core/benchmark/BASELINE-0.1.0.md`.

| Operation | n | 0.1.0 | 1.0.0 |
|---|---:|---:|---:|
| `solve` | 256 | 2.05 | 0.99 |
| `solve` | 1024 | 37.66 | 11.50 |
| `solve` | 2048 | 179.82 | 55.84 |
| `matMat` | 256 | 11.73 | 0.18 |
| `matMat` | 1024 | 37.26 | 5.98 |
| `matMat` | 2048 | 210.71 | 40.82 |
| `matVec` | 1024 | 3.72 | 0.13 |
| `matTransVec` | 1024 | 7.43 | 0.15 |
| `atWa` | 1024 | 42.68 | 6.11 |
| `conditionInf` | 256 | 294.35 | 1.45 |
| `conditionInf` | 1024 | not measured | 24.43 |
| `conditionEstimate` | 1024 | not available | 8.44 |
| `symmetricEigenvalues` | 256 | 90.21 | 3.05 |
| `symmetricEigenvalues` | 1024 | not measured | 155.43 |

In version 0.1.0 the data was converted to an intermediate format on every call,
`conditionInf` and `symmetricEigenvalues` were computed by JVM code, and the condition
estimate from the LU factorization did not exist. In version 1.0.0 the matrix is stored in
the BLAS/LAPACK format and passed without reordering, all `O(n³)` operations are performed by
LAPACK, and `conditionEstimate` computes the condition estimate in `O(n²)` after the
factorization.

## Portable Java implementation

When no native implementation is available, the Java BLAS/LAPACK implementation from netlib
(F2J) is used. It is single-threaded and 10–20 times slower: `solve` at `n = 1024` takes
about 140 ms versus 10–12 ms, `matMat` — about 200 ms versus 6 ms. The library reports the
fall-back to the portable implementation with a warning; `Backends.describe()` returns a
description of the selected implementation. The mode is set by the system property
`-Dnumerics.backend=native|java|auto`.

## OpenBLAS module

The `numerical-core-openblas` module ships OpenBLAS via JavaCPP Presets for systems without
a native BLAS/LAPACK implementation. By default `OpenBlas.install()` sets `min(number of cores, 4)`
threads: on processors with heterogeneous cores a larger thread count slows factorizations
down. With `threads > 1` the calling thread needs a stack of at least 4 MB: either start the
JVM with `-Xss4m`, or solve large systems in a thread with an explicitly specified stack
size; otherwise the JVM may crash inside OpenBLAS. Details are given in `openblas/README.md`.

## Reproducing the measurements

    ./gradlew :numerical-core:benchmark -Pbench.args="256 1024 2048"

The benchmark source code is in `core/src/benchmark/kotlin/numerics/bench/Bench.kt`
(a separate source set, not included in the artifact). To compare implementations, add
`-Dnumerics.backend=java` or `-Dnumerics.backend=native` to the command.
