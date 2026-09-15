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
