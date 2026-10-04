# Sources of the numerical methods

This document records the provenance of every algorithm in the library. The methods used are
classical; references point to the editions against which the formulas were checked and to
the LAPACK documentation for the routines called through netlib.

## Status legend

- **Classical** — the method is described in textbooks or in the LAPACK documentation; the
  implementation is checked against the formulas of the source and by tests.
- **Original work** — an engineering decision of the library without an external source; the
  justification is given in the KDoc of the referenced element and in `docs/ACCURACY.md`.

## 1. Quadrature

| Element | Implementation | Source | Status |
|---|---|---|---|
| Gauss–Legendre nodes and weights on `[-1, 1]`: zeros of the Legendre polynomial `P_m` by Newton's method from the initial guess `cos(π (i + 3/4)/(m + 1/2))`, weights `2 / ((1 − x²) P_m'(x)²)` | `GaussLegendre.gaussLegendreReference` | [Davis, Rabinowitz 1984], ch. 2; [Stoer, Bulirsch 2002], §3.6 | Classical |
| Composite quadrature over a partition of the interval with affine mapping of the nodes onto each subinterval | `GaussLegendre.integrate`, `integrateInterval` | [Davis, Rabinowitz 1984], ch. 2 | Classical |
| Exactness on polynomials of degree `2m − 1` | verified by the quadrature tests | [Stoer, Bulirsch 2002], Theorem 3.6.12 | Classical |
| Gauss–Jacobi nodes and weights for the weight `(1 − x)^a (1 + x)^b` on `[−1, 1]`: Jacobi matrix from the three-term recurrence of the monic Jacobi polynomials, its eigenvalues by Sturm-sequence bisection, Newton polish on the orthonormal polynomial `p̂_m`, Christoffel weights `μ₀ / Σ_{k<m} p̂_k(x_i)²` with `μ₀ = 2^(a+b+1) B(a + 1, b + 1)` | `GaussJacobi`, `GaussJacobi.gaussJacobiReference`, `GaussJacobi.integrate` | [Golub, Welsch 1969]; [DLMF], §18.9 (recurrence), §3.5 (Gauss–Jacobi formula); Sturm-sequence bisection: [Golub, Van Loan 2013], ch. 8 | Classical |
| Product integration of `∫ |t − s|^(−α) f(s) ds`, `0 ≤ α < 1`: on a cell containing `t` the singular factor is absorbed into the weights of the Gauss–Jacobi rules with the weights `(t − s)^(−α)` on `[c0, t]` and `(s − t)^(−α)` on `[t, c1]` | `AlgebraicSingularQuadrature.integrate` | [Golub, Welsch 1969]; [DLMF], §3.5 (Gauss–Jacobi formula) | Classical |
| Geometric refinement of a cell toward a nearby singular point `t` outside it (pieces at distances `d, 2d, 4d, …` from `t`), Gauss–Legendre rule on each piece | `AlgebraicSingularQuadrature.integrate`, `AlgebraicSingularQuadrature.MAX_PIECES` | to be supplied | Classical |

## 2. Dense linear algebra

| Element | Implementation | Source | Status |
|---|---|---|---|
| Solution of a system of linear algebraic equations by LU factorization with partial pivoting | `LinearAlgebra.solve` → LAPACK `dgesv`, `dgetrf`, `dgetrs` | [Anderson et al. 1999]; [Golub, Van Loan 2013], §3.4 | Classical |
| Cholesky factorization and positive definiteness check | `LinearAlgebra.cholesky` → LAPACK `dpotrf` | [Anderson et al. 1999]; [Golub, Van Loan 2013], §4.2 | Classical |
| Matrix–vector and matrix–matrix products, `AᵀWA`, `axpy` | `LinearAlgebra.matVec`, `matTransVec`, `matMat`, `atWa`, `addScaled` → BLAS `dgemv`, `dgemm`, `daxpy` | [Anderson et al. 1999], BLAS appendix | Classical |
| Matrix norms `‖·‖₁`, `‖·‖∞`, Frobenius norm, maximum absolute element | `LinAlgBackend.norm` → LAPACK `dlange` | [Anderson et al. 1999] | Classical |
| Solution check by the backward error `‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)` and the `1e-10` threshold | `LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE` | backward stability of LU factorization: [Higham 2002], ch. 9; justification of the threshold — `docs/ACCURACY.md` | Original work |
| Independent LU factorization implementation in the tests as a second oracle | `core/src/test` | [Golub, Van Loan 2013], §3.4 | Classical |

## 3. Conditioning and error estimation

| Element | Implementation | Source | Status |
|---|---|---|---|
| Backward error `ω` and forward error bound `‖x − x*‖/‖x*‖ ≤ cond∞(A) · ω` | `Conditioning.relativeBackwardError`, `Conditioning.forwardError` | [Higham 2002], ch. 7 (normwise backward error of Rigal–Gaches) | Classical |
| `cond∞(A) = ‖A‖∞ ‖A⁻¹‖∞` via explicit inversion; residual `‖A A⁻¹ − I‖∞` as the reliability indicator | `Conditioning.conditionInf`, `inverse`, `inversionResidual` → LAPACK `dgetrf` + `dgetrs` with the identity matrix as the right-hand side | [Golub, Van Loan 2013], §2.6; reliability criterion — `ConditionEstimate` | Classical / original work |
| Condition number estimate in the 1-norm from the LU factorization | `Conditioning.conditionEstimate` → LAPACK `dgecon` | [Anderson et al. 1999]; [Higham 2002], ch. 15 (Hager–Higham algorithm) | Classical |
| Eigenvalues of a symmetric matrix; spectral condition number | `Conditioning.symmetricEigenvalues`, `conditionSymmetric` → LAPACK `dsyev` | [Anderson et al. 1999]; [Golub, Van Loan 2013], ch. 8 | Classical |
| Sealed type `ForwardError` with variants `Bounded` / `NoFiniteBound` / `Unreliable` | `numerics.ForwardError` | — | Original work: an unreliable estimate is never returned as a number |

## 4. Infrastructure

| Element | Implementation | Source | Status |
|---|---|---|---|
| Noise threshold `1e-13` and the `Measured` type | `numerics.Measured`, `measured` | justification in `docs/ACCURACY.md` | Original work |
| Observed convergence order `log₂(E_h / E_{h/2})` and constant `E_h / h^p` | `orders`, `constCh`, `reliableOrders`, `reliableConstCh` | standard definition of the observed order | Classical |
| Parallel row-wise matrix assembly on the common thread pool | `ParallelAssembly` | — | Original work |
| Unified interface to BLAS/LAPACK implementations with automatic selection and explicit diagnostics | `numerics.backend.LinAlgBackend`, `Backends`, `NetlibBackend` | access through netlib | Original work |
| Bundled OpenBLAS implementation | module `numerical-core-openblas`, `OpenBlas.install()` | OpenBLAS; JavaCPP Presets | Classical |

## 5. Special functions

| Element | Implementation | Source | Status |
|---|---|---|---|
| `Γ(x)`, `ln Γ(x)`: Stirling series with 8 terms (coefficients `B_2k/(2k(2k − 1))`, `k = 1..8`) for `x ≥ 10`; below 10 the recurrence `Γ(x + 1) = x Γ(x)` up to `x + k ≥ 10` | `SpecialFunctions.gamma`, `SpecialFunctions.lnGamma` | [DLMF], §5.11 (Stirling series), §5.5 (recurrence) | Classical |
| Compensated recurrence: the rounding errors of the shifted arguments `x + j` computed exactly (TwoSum) and compensated to first order, `Γ(y + e) ≈ Γ(y)(1 + ψ(y)e)` | `SpecialFunctions.gamma`, `SpecialFunctions.lnGamma` | to be supplied | Classical |
| `B(p, q) = Γ(p)·(Γ(q)/Γ(p + q))` for `p + q ≤ 170` with a first-order correction for the rounding of `p + q`; `exp(ln Γ(p) + ln Γ(q) − ln Γ(p + q))` above | `SpecialFunctions.beta` | [DLMF], §5.12 | Classical |
| `erfc(x)`: Maclaurin series of `erf` for `|x| < 1`; for `x ≥ 1` the Legendre continued fraction of `Γ(1/2, x²)` evaluated backward with `12 + 160/x²` levels; `exp(−x²)` computed as `exp(−s²)·exp(−(x − s)(x + s))` with `s = ⌊16x⌋/16` (Cody's split); `erfc(x) = 2 − erfc(−x)` for `x ≤ −1` | `SpecialFunctions.erfc` | [DLMF], §7.6 (series), §7.9 and §8.9 (continued fractions); split of `exp(−x²)`: routine `CALERF` of [Cody 1993], which implements [Cody 1969] | Classical |
| Mittag-Leffler function `E_β(z) = Σ_{k≥0} z^k / Γ(βk + 1)`: the defining power series (positive terms, no cancellation) | `SpecialFunctions.mittagLeffler` | [DLMF], §10.46 | Classical |

## References

1. **[Davis, Rabinowitz 1984]** Davis P. J., Rabinowitz P. Methods of Numerical
   Integration. — 2nd ed. — Orlando: Academic Press, 1984.
2. **[Stoer, Bulirsch 2002]** Stoer J., Bulirsch R. Introduction to Numerical Analysis. —
   3rd ed. — New York: Springer, 2002. — (Texts in Applied Mathematics; vol. 12).
3. **[Golub, Van Loan 2013]** Golub G. H., Van Loan C. F. Matrix Computations. — 4th ed. —
   Baltimore: Johns Hopkins University Press, 2013.
4. **[Higham 2002]** Higham N. J. Accuracy and Stability of Numerical Algorithms. —
   2nd ed. — Philadelphia: SIAM, 2002.
5. **[Anderson et al. 1999]** Anderson E., Bai Z., Bischof C., Blackford S., Demmel J.,
   Dongarra J., Du Croz J., Greenbaum A., Hammarling S., McKenney A., Sorensen D.
   LAPACK Users' Guide. — 3rd ed. — Philadelphia: SIAM, 1999.
6. **netlib** — `dev.ludovic.netlib` 3.2.0, a JVM access layer to BLAS/LAPACK with a portable
   F2J implementation. MIT License. <https://github.com/luhenry/netlib>.
7. **OpenBLAS** — an optimized BLAS/LAPACK implementation. BSD-3-Clause License.
   <https://github.com/OpenMathLib/OpenBLAS>. Shipped via JavaCPP Presets
   (`org.bytedeco:openblas`, Apache-2.0 / GPLv2 with classpath exception license).
8. **[Golub, Welsch 1969]** Golub G. H., Welsch J. H. Calculation of Gauss quadrature rules //
   Mathematics of Computation. — 1969. — Vol. 23, no. 106. — P. 221–230. —
   DOI: 10.1090/S0025-5718-69-99647-1.
9. **[Cody 1969]** Cody W. J. Rational Chebyshev approximations for the error function //
   Mathematics of Computation. — 1969. — Vol. 23, no. 107. — P. 631–637. —
   DOI: 10.1090/S0025-5718-1969-0247736-4.
10. **[Cody 1993]** Cody W. J. Algorithm 715: SPECFUN–a portable FORTRAN package of special
    function routines and test drivers // ACM Transactions on Mathematical Software. — 1993. —
    Vol. 19, no. 1. — P. 22–30. — DOI: 10.1145/151271.151273. Routine `CALERF`:
    <https://www.netlib.org/specfun/erf>.
11. **[DLMF]** NIST Digital Library of Mathematical Functions. <https://dlmf.nist.gov/>,
    Version 1.2.8, release date 2026-09-15 (accessed 2026-10-04). Sections used:
    §3.5 Quadrature (<https://dlmf.nist.gov/3.5>), §5.5 Functional Relations
    (<https://dlmf.nist.gov/5.5>), §5.11 Asymptotic Expansions (<https://dlmf.nist.gov/5.11>),
    §5.12 Beta Function (<https://dlmf.nist.gov/5.12>), §7.6 Series Expansions
    (<https://dlmf.nist.gov/7.6>), §7.9 Continued Fractions (<https://dlmf.nist.gov/7.9>),
    §8.9 Continued Fractions (<https://dlmf.nist.gov/8.9>), §10.46 Generalized and Incomplete
    Bessel Functions; Mittag-Leffler Function (<https://dlmf.nist.gov/10.46>), §18.9 Recurrence
    Relations and Derivatives (<https://dlmf.nist.gov/18.9>).
