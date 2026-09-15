package numerics

import kotlin.math.abs
import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.backend.MatrixNorm

/**
 * Linear algebra over dense matrices — the library's single entry point.
 * The primary matrix type is [DenseMatrix] (flat column-major layout); the
 * [Array]<[DoubleArray]> overloads are kept as adapters that copy via [DenseMatrix.fromRows]
 * and [DenseMatrix.toRows].
 *
 * Computational operations (products, linear solves, Cholesky factorization) are performed
 * by the [LinAlgBackend] implementation passed as the last parameter; the default is
 * [Backends.default] (system BLAS/LAPACK when available, otherwise a pure-Java
 * implementation). The backend is passed as a parameter rather than a global switch:
 * the result of a call does not depend on state set by other code in the same JVM.
 * Calling [Backends.default] allocates no objects and is suitable for use in a loop.
 * Vector norms, the asymmetry measure and constructors are implemented in [DenseOps]
 * without going through the backend.
 *
 * Shape-consistency and non-emptiness checks are performed here ([Shapes]), before the
 * computation is handed to the backend, and fail with an [IllegalArgumentException]
 * carrying a descriptive message.
 * Non-finite values (NaN, infinity) are checked only in [solve] and [cholesky];
 * all other operations propagate them according to floating-point arithmetic rules, like BLAS.
 */
public object LinearAlgebra {

    // --- Matrix constructors (no backend involved) ----------------------------

    /** Creates a zero matrix of size rows x cols. */
    public fun zeros(rows: Int, cols: Int): Array<DoubleArray> = DenseOps.zeros(rows, cols)

    /** Identity matrix of size n x n. */
    public fun identity(n: Int): Array<DoubleArray> = DenseOps.identity(n)

    /** Creates a zero matrix rows x cols in flat column-major layout. */
    public fun zerosMatrix(rows: Int, cols: Int): DenseMatrix = DenseMatrix.zeros(rows, cols)

    /** Identity matrix n x n in flat column-major layout. */
    public fun identityMatrix(n: Int): DenseMatrix = DenseMatrix.identity(n)

    // --- Operations on DenseMatrix: computation is delegated to the backend ---

    /** Product of matrix A (m x k) and vector x (k) -> vector (m). */
    public fun matVec(
        a: DenseMatrix,
        x: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
        Shapes.requireNonEmpty(a, "matrix A")
        Shapes.requireVectorLength(x, a.cols, "x")
        return backend.matVec(a, x)
    }

    /** Transposed product A^T y, A: m x n, y: m -> vector n. */
    public fun matTransVec(
        a: DenseMatrix,
        y: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
        Shapes.requireNonEmpty(a, "matrix A")
        Shapes.requireVectorLength(y, a.rows, "y")
        return backend.matTransVec(a, y)
    }

    /** Product of matrices A (m x k) and B (k x p) -> (m x p). */
    public fun matMat(
        a: DenseMatrix,
        b: DenseMatrix,
        backend: LinAlgBackend = Backends.default(),
    ): DenseMatrix {
        Shapes.requireMultiplicable(a, b)
        return backend.matMat(a, b)
    }

    /** Product A^T diag(w) A for A: m x n, w: m -> symmetric n x n. */
    public fun atWa(
        a: DenseMatrix,
        w: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DenseMatrix {
        Shapes.requireNonEmpty(a, "matrix A")
        Shapes.requireVectorLength(w, a.rows, "w")
        // B = diag(w)·A: row i is scaled by w[i]; then A^T·B in a single kernel call.
        val scaled = a.copy()
        val d = scaled.data
        val m = a.rows
        for (j in 0 until a.cols) {
            val base = j * m
            for (i in 0 until m) d[base + i] *= w[i]
        }
        return backend.matTransMat(a, scaled)
    }

    /** Element-wise sum A + s*B (matrices of the same shape). */
    public fun addScaled(
        a: DenseMatrix,
        b: DenseMatrix,
        s: Double,
        backend: LinAlgBackend = Backends.default(),
    ): DenseMatrix {
        Shapes.requireSameShape(a, b)
        val y = a.data.copyOf()
        backend.axpy(s, b.data, y)
        return DenseMatrix.fromColumnMajor(a.rows, a.cols, y)
    }

    /**
     * Relative residual tolerance for linear solves, shared by all backends.
     *
     * Criterion: `||Ax-b||_inf <= SINGULARITY_RELATIVE_TOLERANCE * max(||A||_inf*||x||_inf, ||b||_inf)`.
     *
     * The criterion is the residual, not the condition number. The residual is available for
     * any backend and is cheap: O(n²) versus O(n³) for the solve itself. A condition-number
     * check would be wrong in principle: systems with condition number around 1e10 are solved
     * backward-stably (residual near 1e-16 relative to the scale) and must not be rejected.
     *
     * The 1e-10 threshold is justified by the backward stability of LU with partial pivoting:
     * the residual of a correct solution is bounded by `c·n·ε·||A||·||x||` with
     * ε = 2.2e-16 and a moderate growth factor c, i.e. roughly `n·ε` relative to the scale
     * of the system. For n up to 10⁴ the upper bound is about 1e-12, and the observed
     * residual is usually several orders of magnitude below the bound because rounding
     * errors partially cancel. The 1e-10 threshold leaves a margin of at least two orders
     * of magnitude over the upper bound and four to five over the typical residual: the check
     * rejects a solution that does not satisfy the system, rather than distinguishing shades
     * of quality between two correct factorizations.
     *
     * The `||b||_inf` term under the maximum protects the case of nearly zero x: the product
     * `||A||*||x||` is then close to zero, and without this term any rounding error would
     * look like singularity.
     *
     * Scope: the check guarantees that the method does not return a non-finite solution or
     * one that fails to satisfy the system; it does not measure the forward error — use
     * [solveDiagnosed] for that.
     */
    public const val SINGULARITY_RELATIVE_TOLERANCE: Double = 1e-10

    /**
     * Solves the dense linear system A x = b with the given backend.
     *
     * The inputs A and b are not modified. Uniform singularity semantics are enforced
     * here: non-finite input (NaN or infinity) is rejected first, then the backend
     * reports a zero pivot, and finally the residual of the solution is checked
     * (see [SINGULARITY_RELATIVE_TOLERANCE]).
     *
     * The post-check controls the backward error, not the accuracy of the result: on a
     * numerically singular matrix the residual stays small while the forward error
     * grows like `cond(A) · ε`. The forward error is not measured here to keep the
     * solve cheap; [solveDiagnosed] estimates it, and the difference between the two
     * errors is described in [ForwardError].
     *
     * @return solution vector x of length n.
     * @throws IllegalArgumentException if A is empty or non-square, or the length of b does not match.
     * @throws IllegalStateException on non-finite input, singularity, or a residual
     *         inconsistent with machine precision.
     */
    public fun solve(
        a: DenseMatrix,
        b: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
        Shapes.requireSquare(a, "matrix A")
        Shapes.requireVectorLength(b, a.rows, "b")
        for (v in a.data) if (!v.isFinite()) error("System contains non-finite values (NaN or infinity)")
        for (v in b) if (!v.isFinite()) error("System contains non-finite values (NaN or infinity)")
        val n = a.rows
        val x = backend.solve(a, DenseMatrix.fromColumnMajor(n, 1, b.copyOf())).data
        checkSolution(a, b, x, backend)
        return x
    }

    /**
     * Solution of a linear system together with an estimate of its forward error.
     *
     * @property x the same vector that [LinearAlgebra.solve] would return on the same inputs
     *   — the diagnostics neither change the computation nor refine the solution.
     * @property forwardError bound on the relative forward error, or explicit evidence
     *   that no bound exists; see [ForwardError].
     */
    public class DiagnosedSolution(public val x: DoubleArray, public val forwardError: ForwardError)

    /**
     * Solves `A x = b` and estimates the forward error of the computed solution.
     *
     * The post-check inside [solve] controls the backward error — the residual relative
     * to the scale of the system; for LU with partial pivoting it is always small, including
     * on numerically singular matrices, and says nothing about accuracy. The forward error
     * grows like `cond(A) · ε` and eats significant digits on an ill-conditioned system.
     *
     * The diagnostics are not built into [solve] because they cost a separate O(n³)
     * (or O(n²) on top of the factorization with [ConditionSource.ESTIMATE]) and are not
     * needed on every solve.
     *
     * The function does not throw for a large `cond` and does not replace the solution;
     * whether to trust the number is up to the caller. On an exactly singular system it throws
     * [IllegalStateException] under the same contract as [solve].
     *
     * @param source how to estimate `cond`; see [ConditionSource].
     * @param tolerance reliability threshold on the inversion residual; affects the result
     *        only with [ConditionSource.INVERSION].
     * @return the solution and an estimate of its forward error.
     * @throws IllegalStateException on singularity — under the same contract as [solve].
     * @throws IllegalArgumentException with [ConditionSource.SYMMETRIC_SPECTRUM] and a non-symmetric `A`.
     */
    public fun solveDiagnosed(
        a: DenseMatrix,
        b: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
        source: ConditionSource = ConditionSource.INVERSION,
        tolerance: Double = Conditioning.INVERSION_RESIDUAL_TOLERANCE,
    ): DiagnosedSolution {
        val x = solve(a, b, backend)
        val omega = Conditioning.relativeBackwardError(a, b, x, backend)
        val forward = when (source) {
            ConditionSource.INVERSION ->
                Conditioning.forwardError(Conditioning.conditionInf(a, tolerance, backend), omega)
            ConditionSource.ESTIMATE ->
                Conditioning.forwardError(Conditioning.conditionEstimate(a, tolerance, backend), omega)
            ConditionSource.SYMMETRIC_SPECTRUM -> Conditioning.forwardErrorSymmetric(a, omega, backend)
        }
        return DiagnosedSolution(x, forward)
    }

    /** Post-check of the solution, shared by all backends (see [SINGULARITY_RELATIVE_TOLERANCE]). */
    private fun checkSolution(a: DenseMatrix, b: DoubleArray, x: DoubleArray, backend: LinAlgBackend) {
        val n = a.rows
        for (i in x.indices) {
            if (x[i].isNaN() || x[i].isInfinite()) {
                error("solve: matrix is singular — non-finite solution x[$i]=${x[i]} (n=$n)")
            }
        }
        val ax = backend.matVec(a, x)
        var residual = 0.0
        for (i in 0 until n) residual = maxOf(residual, abs(ax[i] - b[i]))
        var matrixNorm = 0.0
        val d = a.data
        for (i in 0 until n) {
            var rowSum = 0.0
            for (j in 0 until n) rowSum += abs(d[i + j * n])
            matrixNorm = maxOf(matrixNorm, rowSum)
        }
        val solutionNorm = normInf(x)
        val rhsNorm = normInf(b)
        val scale = maxOf(matrixNorm * solutionNorm, rhsNorm)
        if (residual > SINGULARITY_RELATIVE_TOLERANCE * scale) {
            error(
                "solve: matrix is singular — residual too large: n=$n, " +
                    "||A||_inf=$matrixNorm, ||x||_inf=$solutionNorm, ||b||_inf=$rhsNorm, " +
                    "||Ax-b||_inf=$residual > $SINGULARITY_RELATIVE_TOLERANCE * $scale"
            )
        }
    }

    // --- Scalar and utility operations (no backend involved) ------------------

    /** Euclidean norm of a vector. */
    public fun norm2(x: DoubleArray): Double = DenseOps.norm2(x)

    /** Infinity (uniform) norm of a vector. */
    public fun normInf(x: DoubleArray): Double = DenseOps.normInf(x)

    /**
     * Cholesky factorization A = L Lᵀ of a symmetric positive definite A (dpotrf).
     *
     * @return lower-triangular L (entries above the diagonal are zero), or `null`
     *   if A is not positive definite.
     * @throws IllegalArgumentException if A is empty, non-square, contains non-finite
     *   values, or its asymmetry exceeds `1e-12 · ‖A‖∞`.
     */
    public fun cholesky(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): DenseMatrix? {
        Shapes.requireSquare(a, "matrix A")
        Shapes.requireFinite(a, "matrix A")
        Shapes.requireSymmetric(a, backend.norm(a, MatrixNorm.INF), "matrix A")
        return backend.cholesky(a)
    }

    /** Asymmetry: max|A - A^T|; requires a non-empty square A. */
    public fun maxAsymmetry(a: DenseMatrix): Double {
        Shapes.requireSquare(a, "matrix A")
        return DenseOps.maxAsymmetry(a.toRows())
    }

    // --- Adapters over Array<DoubleArray> ---------------------------------------
    // A ragged array is rejected by DenseMatrix.fromRows (IllegalArgumentException with the row index).

    private fun rows(a: Array<DoubleArray>, op: String): DenseMatrix {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "$op: matrix A is empty" }
        return DenseMatrix.fromRows(a)
    }

    /** Product of matrix A (m x k) and vector x (k) -> vector (m). */
    public fun matVec(a: Array<DoubleArray>, x: DoubleArray, backend: LinAlgBackend = Backends.default()): DoubleArray =
        matVec(rows(a, "matVec"), x, backend)

    /** Transposed product A^T y, A: m x n, y: m -> vector n. */
    public fun matTransVec(a: Array<DoubleArray>, y: DoubleArray, backend: LinAlgBackend = Backends.default()): DoubleArray =
        matTransVec(rows(a, "matTransVec"), y, backend)

    /** Product of matrices A (m x k) and B (k x p) -> (m x p). */
    @Deprecated(
        "Use the DenseMatrix overload",
        ReplaceWith("matMat(DenseMatrix.fromRows(a), DenseMatrix.fromRows(b), backend)"),
        DeprecationLevel.WARNING,
    )
    public fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>, backend: LinAlgBackend = Backends.default()): Array<DoubleArray> {
        require(b.isNotEmpty() && b[0].isNotEmpty()) { "matMat: matrix B is empty" }
        return matMat(rows(a, "matMat"), DenseMatrix.fromRows(b), backend).toRows()
    }

    /** Product A^T diag(w) A for A: m x n, w: m -> symmetric n x n. */
    @Deprecated(
        "Use the DenseMatrix overload",
        ReplaceWith("atWa(DenseMatrix.fromRows(a), w, backend)"),
        DeprecationLevel.WARNING,
    )
    public fun atWa(a: Array<DoubleArray>, w: DoubleArray, backend: LinAlgBackend = Backends.default()): Array<DoubleArray> =
        atWa(rows(a, "atWa"), w, backend).toRows()

    /** Element-wise sum A + s*B (matrices of the same shape). */
    @Deprecated(
        "Use the DenseMatrix overload",
        ReplaceWith("addScaled(DenseMatrix.fromRows(a), DenseMatrix.fromRows(b), s, backend)"),
        DeprecationLevel.WARNING,
    )
    public fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double, backend: LinAlgBackend = Backends.default()): Array<DoubleArray> {
        require(a.size == b.size) { "addScaled: row counts of A(${a.size}) and B(${b.size}) do not match" }
        return addScaled(DenseMatrix.fromRows(a), DenseMatrix.fromRows(b), s, backend).toRows()
    }

    /**
     * Solves the dense linear system A x = b; same contract as the [DenseMatrix] overload,
     * including the non-finite input check and the residual post-check.
     * @throws IllegalStateException on singularity.
     */
    public fun solve(a: Array<DoubleArray>, b: DoubleArray, backend: LinAlgBackend = Backends.default()): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "solve: matrix A is empty" }
        require(a[0].size == a.size) { "solve: matrix A must be square, got ${a.size}x${a[0].size}" }
        for (i in a.indices) require(a[i].size == a.size) {
            "solve: matrix A is ragged — row $i has length ${a[i].size}, expected ${a.size}"
        }
        return solve(DenseMatrix.fromRows(a), b, backend)
    }

    /** Solves A x = b and estimates the forward error; same contract as the [DenseMatrix] overload. */
    public fun solveDiagnosed(
        a: Array<DoubleArray>,
        b: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
        source: ConditionSource = ConditionSource.INVERSION,
        tolerance: Double = Conditioning.INVERSION_RESIDUAL_TOLERANCE,
    ): DiagnosedSolution = solveDiagnosed(rows(a, "solveDiagnosed"), b, backend, source, tolerance)

    /**
     * Cholesky factorization over an array of rows; same contract as the [DenseMatrix] overload.
     * @return lower-triangular L, or null if A is not positive definite.
     */
    @Deprecated(
        "Use the DenseMatrix overload",
        ReplaceWith("cholesky(DenseMatrix.fromRows(a))"),
        DeprecationLevel.WARNING,
    )
    public fun cholesky(a: Array<DoubleArray>): Array<DoubleArray>? = cholesky(rows(a, "cholesky"))?.toRows()

    /** Asymmetry: max|A - A^T|; requires a non-empty, square, non-ragged A. */
    public fun maxAsymmetry(a: Array<DoubleArray>): Double {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "maxAsymmetry: matrix A is empty" }
        require(a[0].size == a.size) { "maxAsymmetry: matrix A must be square, got ${a.size}x${a[0].size}" }
        for (i in a.indices) require(a[i].size == a.size) {
            "maxAsymmetry: matrix A is ragged — row $i has length ${a[i].size}, expected ${a.size}"
        }
        return DenseOps.maxAsymmetry(a)
    }

    // --- NumericsContext overloads: backend taken from the context ------------

    /** Same as [matVec], with the backend taken from [context]. */
    public fun matVec(a: DenseMatrix, x: DoubleArray, context: NumericsContext): DoubleArray =
        matVec(a, x, context.backend)

    /** Same as [matTransVec], with the backend taken from [context]. */
    public fun matTransVec(a: DenseMatrix, y: DoubleArray, context: NumericsContext): DoubleArray =
        matTransVec(a, y, context.backend)

    /** Same as [matMat], with the backend taken from [context]. */
    public fun matMat(a: DenseMatrix, b: DenseMatrix, context: NumericsContext): DenseMatrix =
        matMat(a, b, context.backend)

    /** Same as [atWa], with the backend taken from [context]. */
    public fun atWa(a: DenseMatrix, w: DoubleArray, context: NumericsContext): DenseMatrix =
        atWa(a, w, context.backend)

    /** Same as [addScaled], with the backend taken from [context]. */
    public fun addScaled(a: DenseMatrix, b: DenseMatrix, s: Double, context: NumericsContext): DenseMatrix =
        addScaled(a, b, s, context.backend)

    /** Same as [solve], with the backend taken from [context]. */
    public fun solve(a: DenseMatrix, b: DoubleArray, context: NumericsContext): DoubleArray =
        solve(a, b, context.backend)

    /** Same as [solveDiagnosed], with the backend taken from [context]. */
    public fun solveDiagnosed(
        a: DenseMatrix,
        b: DoubleArray,
        context: NumericsContext,
        source: ConditionSource = ConditionSource.INVERSION,
        tolerance: Double = Conditioning.INVERSION_RESIDUAL_TOLERANCE,
    ): DiagnosedSolution = solveDiagnosed(a, b, context.backend, source, tolerance)

    /** Same as [cholesky], with the backend taken from [context]. */
    public fun cholesky(a: DenseMatrix, context: NumericsContext): DenseMatrix? =
        cholesky(a, context.backend)
}
