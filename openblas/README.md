# numerical-core-openblas

## Purpose

This module supplies `numerical-core` with a bundled BLAS/LAPACK implementation — OpenBLAS.
It is intended for machines without a system BLAS/LAPACK implementation: workstations,
laptops, containers built from a minimal image. On compute clusters with MKL, AOCL or
OpenBLAS installed the module is not needed: the library uses the system implementation.

## Setup

```kotlin
implementation("io.github.egorkakulikov:numerical-core-openblas:<version>")
```

The dependency on `numerical-core` is added transitively.

## Usage

Call `OpenBlas.install()` before the first use of the library, because the path
to the native implementation is read once, at initialization:

```kotlin
fun main() {
    OpenBlas.install()
    println(Backends.describe())
}
```

Without this module the same effect is achieved by setting the system properties
`dev.ludovic.netlib.blas.nativeLibPath` and `dev.ludovic.netlib.lapack.nativeLibPath`
to the full path of the library file.

## Threads and stack

By default OpenBLAS uses `min(number of cores, 4)` threads: on processors with
heterogeneous cores a larger thread count slows the factorizations down. The thread count
is set with the `OpenBlas.install(threads = …)` parameter.

With `threads > 1` the calling thread needs a stack of at least 4 MB: start the JVM
with `-Xss4m`, or solve large systems in a thread with an explicit stack size —
`Thread(null, r, "lapack", 8L shl 20)`. Otherwise the JVM may crash inside
OpenBLAS.

## Size and platforms

The dependency adds 13–48 MB per platform. Supported platforms: linux-x86_64, linux-arm64,
macosx-arm64, macosx-x86_64, windows-x86_64.

## Verification

After `OpenBlas.install()` the `Backends.describe()` method returns the description of the
native OpenBLAS implementation.
