# Accuracy and reliability of results

This document describes the quantities the library computes to control accuracy, their
mathematical meaning, and the guarantees provided by each public method. All numerical
examples are taken from the library tests.

## Backward error

The relative backward error of a computed solution `x` of a system of linear algebraic
equations `Ax = b` is the quantity

    ω = ‖Ax − b‖∞ / max(‖A‖∞ ‖x‖∞, ‖b‖∞).

By the Rigal–Gaches theorem, `ω` equals the smallest number for which `x` is the exact
solution of the perturbed system `(A + ΔA) x = b + Δb` with `‖ΔA‖∞ ≤ ω ‖A‖∞` and `‖Δb‖∞ ≤ ω ‖b‖∞`.
Thus `ω` characterizes the smallest relative perturbation of the input data for which the
computed solution becomes exact.

LU factorization with partial pivoting is backward stable: for a correct implementation `ω`
is of order `n·ε`, where `ε ≈ 1.1·10⁻¹⁶` is the unit roundoff of double precision.
Checking `ω` therefore detects any solver failure (a singular matrix, non-numeric data, a
defect in the LAPACK implementation) regardless of which library performed the
factorization. The quantity `ω` costs `O(n²)` operations and does not require knowledge of
the exact solution.

`LinearAlgebra.solve` computes `ω` after every solve and throws an exception if `ω` exceeds
`SINGULARITY_RELATIVE_TOLERANCE = 1e-10` or the solution contains non-numeric values. For
non-numeric input (`NaN`, `±Inf`) the exception is thrown before LAPACK is called.

### Justification of the 1e-10 threshold

For matrices of size up to `10⁴` the expected backward error of a correct solution does not
exceed `n·ε ≈ 10⁻¹²`, and in practice it is two to three orders of magnitude smaller. The
threshold `10⁻¹⁰` leaves a margin of two orders of magnitude above the worst-case estimate
while remaining many orders of magnitude below the values that arise from numerical
singularity: in that case `ω` typically lies in the range `10⁻³…1`, because the
factorization divides by pivots at noise level. The threshold does not depend on the
conditioning of the matrix: an ill-conditioned but non-singular system is solved with small
`ω`, and the inaccuracy in that case shows up in the forward error, not in the residual.

## Forward error and `ForwardError` modes

The forward error is the distance from the computed solution to the exact solution `x*`.
To first order it is bounded in terms of the backward error by

    ‖x − x*‖∞ / ‖x*‖∞ ≤ cond∞(A) · ω,   cond∞(A) = ‖A‖∞ ‖A⁻¹‖∞.

`LinearAlgebra.solveDiagnosed` returns the solution together with a value of the sealed type
`ForwardError`, which has three variants:

| Variant | Condition | Meaning |
|---|---|---|
| `Bounded(backwardError, cond, relativeBound)` | the `cond` estimate is reliable and `cond · ω < 1` | The relative error does not exceed `relativeBound = cond · ω`; `survivingDigitsOrNull()` returns the number of surviving decimal digits. |
| `NoFiniteBound(backwardError)` | the `cond` estimate is reliable but `cond · ω ≥ 1` | The bound carries no information: the system is numerically singular at the accuracy level of the solution. |
| `Unreliable(backwardError, condition)` | the `cond` estimate is unreliable | The forward error is not estimated; the `condition` field holds the data on which the decision was based. |

An unreliable estimate is never returned as a number: `relativeBoundOrNull()` returns
`null` for the last two variants, and code that needs the bound must handle its absence
explicitly.

```kotlin
val d = LinearAlgebra.solveDiagnosed(a, b)
when (val e = d.forwardError) {
    is ForwardError.Bounded       -> println("error ≤ ${e.relativeBound}")
    is ForwardError.NoFiniteBound -> println("numerically singular system, ω = ${e.backwardError}")
    is ForwardError.Unreliable    -> println("condition number not reliably determined")
}
```

## Three ways to compute the condition number

The `ConditionSource` enumeration selects how `cond` is computed for the forward error
estimate.

**`INVERSION`** — `Conditioning.conditionInf`. The exact value `‖A‖∞ ‖A⁻¹‖∞` is computed via
explicit matrix inversion (LAPACK `dgetrf`, then `dgetrs` with the identity matrix as the
right-hand side); this costs `O(n³)` operations and `n²` additional memory. The reliability
indicator is the inversion residual `‖A·A⁻¹ − I‖∞`: if it exceeds
`INVERSION_RESIDUAL_TOLERANCE = 1e-8`, the computed inverse is inaccurate and
`ConditionEstimate.isReliable` is `false`. The inversion residual grows as `cond · ε`, so
for `cond ≳ 10⁸` the `INVERSION` source stops producing reliable values and reports this
through the reliability flag instead of returning an arbitrary number.

**`ESTIMATE`** — `Conditioning.conditionEstimate`. Uses the LAPACK `dgecon` estimate in the
1-norm from the already computed LU factorization: `O(n²)` operations after the
factorization, no additional matrix. The estimate is a lower bound for `cond₁(A)` and
differs from the exact value by at most a small factor (for the Hager–Higham algorithm,
typically no more than three). For symmetric matrices `cond₁ = cond∞`; in general the norms
differ by a factor depending on `n`, so the bound `cond · ω` with this source is
approximate.

**`SYMMETRIC_SPECTRUM`** — `Conditioning.conditionSymmetric`. For a symmetric matrix
`cond₂ = |λ|max / |λ|min` is computed from the spectrum obtained with the LAPACK routine
`dsyev`. For a non-symmetric matrix (as measured by `maxAsymmetry`) an exception is thrown.

### Example: the 8×8 Hilbert matrix

For the Hilbert matrix of order 8, `cond∞ ≈ 1.5·10¹⁰`. Two correct LAPACK implementations
(Apple Accelerate and the portable Java implementation) produce solutions of `Hx = b` that
differ by `≈1.6·10⁻⁸`, whereas on well-conditioned matrices the discrepancy does not exceed
`2·10⁻¹⁶`. This discrepancy is expected and stays within the bound
`cond · ω ≈ 1.5·10¹⁰ · 10⁻¹⁶`. The inversion residual on the same matrix is `10⁻⁷…10⁻⁶`
depending on the implementation, which exceeds the `10⁻⁸` threshold; hence the `INVERSION`
source returns `Unreliable`, while the `ESTIMATE` source returns a reliable estimate of
order `10¹⁰`.

## Measured quantities and the noise threshold

For quantities obtained by summation (norms of differences, errors on grids) the library
provides the type `Measured` and the function `measured(value, threshold = MACHINE_NOISE_THRESHOLD)`
with two result variants:

- `Reliable(value)` — `|value| ≥ threshold`: the quantity reflects the phenomenon being measured;
- `AtNoiseLevel(value, threshold)` — `|value| < threshold`: the quantity is indistinguishable
  from accumulated rounding error.

The boundary is inclusive: a value equal to the threshold is considered reliable. The
threshold `MACHINE_NOISE_THRESHOLD = 1e-13` is chosen as follows: summing `10³` terms of
order one accumulates an error up to `10³ · ε ≈ 10⁻¹³`; for longer sums or data of a
different scale the threshold is set explicitly. The functions `reliableOrders` and
`reliableConstCh` compute the observed convergence order `log₂(e_h / e_{h/2})` and the
constant `C = e_h / h^p` only from reliable measurements and return `null` if at least one
of the errors lies at noise level: otherwise the logarithm of the ratio of two noise-level
quantities would be mistaken for a meaningful convergence order.

## Reproducibility on multithreaded implementations

System BLAS/LAPACK implementations with internal multithreading (Apple Accelerate, OpenBLAS,
Intel MKL) may, under concurrent calls from several threads, produce results that differ in
the least significant digits (relative discrepancy of order `10⁻¹⁴`) because of a
non-deterministic summation order. This applies not only to spectrum computation
(`dsyev` in `Conditioning.symmetricEigenvalues`) but also to solving linear systems (`dgesv` in
`LinearAlgebra.solve`) and to matrix multiplication (`dgemm` in `LinearAlgebra.matMat`). The
portable Java implementation is deterministic: identical inputs give bit-identical results
regardless of the number of threads.

The library does not hide this difference. Tests compare the results of system
implementations with a corresponding tolerance (`ConcurrencyTest`, relative accuracy `10⁻¹³`);
bit-level reproducibility is guaranteed only in sequential mode and in `ParallelAssembly`,
where each task writes its result into its own region of memory.

## Summary of guarantees

| Method | Checked condition | Action on violation |
|---|---|---|
| `LinearAlgebra.solve` | `ω ≤ 1e-10`, no `NaN`/`Inf` in input or output | `IllegalStateException` |
| `LinearAlgebra.solveDiagnosed` | the same, plus the forward error estimate | `ForwardError.NoFiniteBound` or `Unreliable` instead of a number |
| `LinearAlgebra.cholesky` | symmetry (`maxAsymmetry`), positive definiteness | `IllegalArgumentException`; `null` for a matrix that is not positive definite |
| `Conditioning.conditionInf` | inversion residual `≤ 1e-8` | `ConditionEstimate.isReliable = false`, `valueOrNull() = null` |
| `Conditioning.conditionEstimate` | singularity of the LU factorization | `isReliable = false` for a singular matrix |
| `Conditioning.symmetricEigenvalues` | symmetry of the input | `IllegalArgumentException` |
| `GaussLegendre.integrate` | at least two partition points, monotone partition | `IllegalArgumentException` |
| `measured`, `reliableOrders`, `reliableConstCh` | `|value| ≥ threshold` | `AtNoiseLevel` / `null` |

All checks are performed identically for every BLAS/LAPACK implementation: the system one,
the bundled OpenBLAS, and the portable Java one.
