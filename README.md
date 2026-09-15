# numerical-core

A dense linear algebra library for the JVM. Computations are carried out by the system BLAS/LAPACK
implementation of the machine the program runs on, and every result comes with an estimate of the number of reliable significant digits.

## Rationale

Existing tools for the JVM solve the problem only in part. LAPACK bindings (netlib, JavaCPP Presets
for OpenBLAS and MKL) expose the `dgesv` routine in its raw form: the return code `info = 0` only indicates
that the factorization completed, not that the solution is accurate. Java libraries (EJML, ojAlgo, Hipparchus)
do not use the system BLAS/LAPACK implementations (MKL, Accelerate) and are therefore
10–20 times slower on matrices of order 10³ and above. None of these tools tells the calling program
how many significant digits of the result are reliable. numerical-core is designed to
remove these shortcomings.

Reliability of the result is encoded in the type system. The `solve` method returns a solution only after checking
the backward error; for a singular system it throws an exception. The `solveDiagnosed` method returns a `ForwardError`
with three outcomes (bound obtained, no bound, estimate unreliable), and the compiler checks that all of them are handled.

Performance is determined by the system BLAS/LAPACK implementation. A matrix is stored in LAPACK column-major order
and passed to the `dgesv` and `dgemm` routines without reordering. On a node with Intel MKL, MKL is used; on
Apple computers, Accelerate; on Linux, the system OpenBLAS; the number of threads is controlled by the cluster scheduler.
Solving a 1024×1024 system of linear algebraic equations takes 10 ms versus 140 ms
for a Java implementation; the condition number is computed in 27 ms versus 24 s for inverting the matrix through n linear solves.

The library behaves identically on every BLAS/LAPACK implementation. Residual checks, exceptions and result formats
are the same for MKL, OpenBLAS, Accelerate and the portable Java implementation used when no native one is available.
Falling back to the portable implementation is reported explicitly; a silent switch to the slow path cannot happen.

Primitives absent from LAPACK are implemented in the same style and with the same guarantees. These include composite
Gauss–Legendre quadrature, parallel matrix assembly with a bit-for-bit reproducible result, and computation of the
observed convergence order using only the measurements that exceed rounding noise.

## Setup

JDK 21 or newer is required. The dependency `io.github.egorkakulikov:numerical-core:1.1.0` is published to GitHub Packages;
reading it requires a token with the `read:packages` scope (`gpr.user` and `gpr.token` in `~/.gradle/gradle.properties` or environment variables):

```kotlin
maven {
    url = uri("https://maven.pkg.github.com/EgorkaKulikov/numerical-core")
    credentials {
        username = providers.gradleProperty("gpr.user").orNull ?: System.getenv("GITHUB_ACTOR")
        password = providers.gradleProperty("gpr.token").orNull ?: System.getenv("GITHUB_TOKEN")
    }
}
```

If the machine has no system BLAS/LAPACK implementation, add the dependency
`io.github.egorkakulikov:numerical-core-openblas:1.1.0` and call `OpenBlas.install()` before the first
use of the library: it unpacks OpenBLAS for the current platform (see `openblas/README.md`).

## Documentation

Reliability thresholds and their justification are described in `docs/ACCURACY.md`; the data path, multithreading and measurements in
`docs/PERFORMANCE.md`; the sources of the algorithms in `docs/SOURCES.md`; the API documentation is built with `./gradlew :numerical-core:dokkaHtml`.

## License

The library is distributed under the Apache License 2.0; the copyright holder is Egor Kulikov.
The license text is in the `LICENSE` file; third-party notices are in the `NOTICE` file.
