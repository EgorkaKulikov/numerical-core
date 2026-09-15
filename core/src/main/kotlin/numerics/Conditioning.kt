package numerics

import kotlin.math.abs
import kotlin.math.log10
import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.backend.MatrixNorm

/**
 * Condition-number estimate of a dense matrix together with a reliability flag.
 *
 * The condition number is itself computed numerically, and on nearly singular matrices
 * its value is unreliable: the forward error of the inversion grows like `cond(A)·ε`, and for
 * `cond` of order `1/ε` no correct digits remain in the columns of `A⁻¹`. The residual
 * `‖A·A⁻¹ − I‖∞` measures exactly this loss, so the estimate is accompanied by it,
 * and reliability is checked via [isReliable].
 *
 * @property condInf the condition-number estimate; [Double.POSITIVE_INFINITY] if
 *   the matrix was found to be singular.
 * @property inversionResidual the inversion residual `‖A·A⁻¹ − I‖∞`; [Double.POSITIVE_INFINITY]
 *   if the inversion failed; `0.0` if the estimate was obtained without inversion
 *   ([Conditioning.conditionEstimate]).
 * @property tolerance the residual threshold below which the estimate is considered reliable; strictly positive.
 */
public data class ConditionEstimate(
    val condInf: Double,
    val inversionResidual: Double,
    val tolerance: Double,
) {
    init {
        require(tolerance > 0.0) { "ConditionEstimate: tolerance must be > 0, got $tolerance" }
    }

    /** Whether the estimate is reliable: the value is finite and the inversion residual does not exceed [tolerance]. */
    val isReliable: Boolean
        get() = condInf.isFinite() && inversionResidual <= tolerance

    /** [condInf] if the estimate is reliable, otherwise `null`. */
    public fun valueOrNull(): Double? = if (isReliable) condInf else null
}

/**
 * How the condition number is obtained when [LinearAlgebra.solveDiagnosed]
 * builds the forward-error bound. The variants differ in guarantees, cost and
 * requirements on the matrix.
 */
public enum class ConditionSource {
    /**
     * Explicit matrix inversion ([Conditioning.conditionInf]): O(n³), no requirements on `A`,
     * reliability is controlled by the inversion residual.
     */
    INVERSION,

    /**
     * Spectrum of a symmetric matrix ([Conditioning.conditionSymmetric]): requires symmetric `A`;
     * distinguishes a singular matrix (`min|λ| = 0`) from a matrix with a large but finite `cond`.
     */
    SYMMETRIC_SPECTRUM,

    /**
     * LAPACK estimate from the LU factorization ([Conditioning.conditionEstimate]): O(n²) on top of
     * the factorization, no requirements on `A`; gives a lower bound within a factor of a few units.
     */
    ESTIMATE,
}

/**
 * Estimate of the relative forward error of a linear-system solution together with a verdict
 * on whether it exists and whether it can be trusted.
 *
 * The backward error `ω = ‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)` shows which system the method
 * solved exactly; for LU with partial pivoting it is small even on a nearly singular matrix.
 * The forward error `‖x − x*‖∞ / ‖x*‖∞` is bounded by the product `cond(A)·ω` and on an
 * ill-conditioned system may consume all significant digits. A small backward error
 * by itself does not imply an accurate result.
 *
 * The three modes are separated by constructors: a number can be obtained only by explicit
 * pattern matching or via [relativeBoundOrNull], which returns `null` where there is no number.
 */
public sealed interface ForwardError {
    /** The measured relative backward error; available in all modes. */
    public val backwardError: Double

    /**
     * The estimate exists and is reliable: `‖x − x*‖∞ / ‖x*‖∞ <= [relativeBound]`.
     *
     * @property cond the condition number used.
     * @property relativeBound the product `cond · backwardError`.
     */
    public data class Bounded(
        override val backwardError: Double,
        val cond: Double,
        val relativeBound: Double,
    ) : ForwardError

    /**
     * There is no finite bound: the matrix is numerically singular — the inversion failed or
     * the spectral estimate produced a zero eigenvalue.
     */
    public data class NoFiniteBound(override val backwardError: Double) : ForwardError

    /**
     * The `cond` estimate is finite but unreliable ([ConditionEstimate.isReliable] is `false`).
     *
     * @property condition the unreliable estimate itself — it carries the inversion residual.
     */
    public data class Unreliable(
        override val backwardError: Double,
        val condition: ConditionEstimate,
    ) : ForwardError

    /** The forward-error bound, or `null` in the [NoFiniteBound] and [Unreliable] modes. */
    public fun relativeBoundOrNull(): Double? = (this as? Bounded)?.relativeBound

    /**
     * How many decimal digits of the result are guaranteed to survive: `−log10(bound)`,
     * clamped to 0 from below and to 16 from above (the full double mantissa).
     * `null` exactly where [relativeBoundOrNull] is `null`.
     */
    public fun survivingDigitsOrNull(): Double? {
        val bound = relativeBoundOrNull() ?: return null
        if (bound <= 0.0) return 16.0
        return minOf(16.0, maxOf(0.0, -log10(bound)))
    }
}

/**
 * Condition numbers of dense matrices and forward-error estimation for linear-system solutions.
 *
 * All computations are performed by a [LinAlgBackend]: inversion — solving A·X = I via the LU factorization (dgetrf + dgetrs),
 * condition estimation — dgecon, spectrum of a symmetric matrix — dsyev. The primary
 * overloads work with [DenseMatrix]; the [Array]<[DoubleArray]> overloads are adapters
 * that copy via [DenseMatrix.fromRows].
 *
 * The estimate via explicit inversion ([conditionInf]) is reliable on well-conditioned
 * matrices and loses meaning on nearly singular ones: `A⁻¹` is computed by the same LU, whose
 * forward error grows like `cond(A)·ε`. The residual `‖A·A⁻¹ − I‖∞` shows this loss
 * directly, so the result always carries it ([ConditionEstimate]). For symmetric
 * matrices the spectral estimate [conditionSymmetric] is more robust: orthogonal transformations
 * do not invert the matrix and honestly yield a zero eigenvalue on a singular one.
 */
public object Conditioning {

    /**
     * Reliability threshold for the [conditionInf] estimate in terms of the inversion residual `‖A·A⁻¹ − I‖∞`.
     * Corresponds to losing half of the significant digits: for a larger residual not even
     * half of the digits of the `cond` estimate are correct.
     */
    public const val INVERSION_RESIDUAL_TOLERANCE: Double = 1e-8

    /** Row-sum matrix norm `‖A‖∞ = max_i Σ_j |a_ij|`; the matrix must be non-empty. */
    public fun matrixNormInf(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): Double {
        Shapes.requireNonEmpty(a, "matrix A")
        return backend.norm(a, MatrixNorm.INF)
    }

    /** Row-sum matrix norm over a row array; see the [DenseMatrix] overload. */
    public fun matrixNormInf(a: Array<DoubleArray>): Double = matrixNormInf(rows(a))

    /**
     * Inverse of a square matrix via the LU factorization.
     *
     * @return `A⁻¹`, or `null` if the matrix was found to be singular or the result contains
     *   non-finite values. Reliability of a finite result is not guaranteed — it is
     *   measured by [inversionResidual].
     */
    public fun inverse(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): DenseMatrix? {
        Shapes.requireSquare(a, "matrix A")
        val inv = backend.inverse(a) ?: return null
        for (v in inv.data) if (!v.isFinite()) return null
        return inv
    }

    /** Inverse over a row array; see the [DenseMatrix] overload. */
    @Deprecated(
        "Use the DenseMatrix overload",
        ReplaceWith("inverse(DenseMatrix.fromRows(a))"),
        DeprecationLevel.WARNING,
    )
    public fun inverse(a: Array<DoubleArray>): Array<DoubleArray>? = inverse(rows(a))?.toRows()

    /** Inversion residual `‖A·B − I‖∞` — a measure of how many digits survived in `B ≈ A⁻¹`. */
    public fun inversionResidual(a: DenseMatrix, inv: DenseMatrix, backend: LinAlgBackend = Backends.default()): Double {
        Shapes.requireSquare(a, "matrix A")
        Shapes.requireSquare(inv, "matrix A⁻¹")
        Shapes.requireSameShape(a, inv)
        val r = backend.matMat(a, inv)
        val n = a.rows
        val d = r.data
        for (i in 0 until n) d[i + i * n] -= 1.0
        return backend.norm(r, MatrixNorm.INF)
    }

    /** Inversion residual over row arrays; see the [DenseMatrix] overload. */
    public fun inversionResidual(a: Array<DoubleArray>, inv: Array<DoubleArray>): Double =
        inversionResidual(rows(a), rows(inv))

    /**
     * Estimate of `cond∞(A) = ‖A‖∞ · ‖A⁻¹‖∞` via explicit inversion, together with the inversion residual.
     *
     * No exception is thrown on a singular matrix: an infinite estimate with an infinite
     * residual is returned, and [ConditionEstimate.isReliable] is `false`.
     *
     * @param tolerance reliability threshold in terms of the inversion residual; strictly positive.
     * @throws IllegalArgumentException if the matrix is empty, not square, or `tolerance <= 0`.
     */
    public fun conditionInf(
        a: DenseMatrix,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
        backend: LinAlgBackend = Backends.default(),
    ): ConditionEstimate {
        require(tolerance > 0.0) { "conditionInf: tolerance must be > 0, got $tolerance" }
        Shapes.requireSquare(a, "matrix A")
        val inv = inverse(a, backend)
            ?: return ConditionEstimate(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, tolerance)
        val cond = matrixNormInf(a, backend) * matrixNormInf(inv, backend)
        return ConditionEstimate(cond, inversionResidual(a, inv, backend), tolerance)
    }

    /** Condition estimate via inversion over a row array; see the [DenseMatrix] overload. */
    public fun conditionInf(
        a: Array<DoubleArray>,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate = conditionInf(rows(a), tolerance)

    /**
     * Condition-number estimate in the 1-norm by LAPACK (dgecon) from the LU factorization.
     *
     * Gives a lower bound, usually within a factor of a few units of the exact value;
     * the cost is O(n²) after the factorization versus O(n³) for explicit inversion.
     * [ConditionEstimate.inversionResidual] is zero because no inversion is performed,
     * and for a finite estimate [ConditionEstimate.isReliable] is true — reliability here
     * is ensured by the algorithm. On a singular matrix an infinite estimate is returned
     * (its [ConditionEstimate.valueOrNull] gives `null`).
     *
     * @throws IllegalArgumentException if the matrix is empty, not square, or `tolerance <= 0`.
     */
    public fun conditionEstimate(
        a: DenseMatrix,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
        backend: LinAlgBackend = Backends.default(),
    ): ConditionEstimate {
        require(tolerance > 0.0) { "conditionEstimate: tolerance must be > 0, got $tolerance" }
        Shapes.requireSquare(a, "matrix A")
        val lu = backend.luFactor(a)
        if (lu.isSingular) return ConditionEstimate(Double.POSITIVE_INFINITY, 0.0, tolerance)
        val rcond = backend.reciprocalCondition1(lu, backend.norm(a, MatrixNorm.ONE))
        val cond = if (rcond > 0.0) 1.0 / rcond else Double.POSITIVE_INFINITY
        return ConditionEstimate(cond, 0.0, tolerance)
    }

    /**
     * Eigenvalues of a symmetric matrix in ascending order (dsyev).
     *
     * @throws IllegalArgumentException if the matrix is empty, not square, or its asymmetry
     *   exceeds `1e-12 · ‖A‖∞`.
     */
    public fun symmetricEigenvalues(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): DoubleArray {
        Shapes.requireSquare(a, "matrix A")
        Shapes.requireSymmetric(a, backend.norm(a, MatrixNorm.INF), "matrix A")
        return backend.symmetricEigenvalues(a)
    }

    /** Eigenvalues over a row array; see the [DenseMatrix] overload. */
    public fun symmetricEigenvalues(a: Array<DoubleArray>): DoubleArray = symmetricEigenvalues(rows(a))

    /**
     * Smallest-magnitude eigenvalue of a symmetric matrix — its `σ_min`.
     * Exactly `0.0` means numerical singularity.
     */
    public fun smallestMagnitudeEigenvalue(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): Double {
        var m = Double.POSITIVE_INFINITY
        for (v in symmetricEigenvalues(a, backend)) m = minOf(m, abs(v))
        return m
    }

    /** Smallest-magnitude eigenvalue over a row array; see the [DenseMatrix] overload. */
    public fun smallestMagnitudeEigenvalue(a: Array<DoubleArray>): Double = smallestMagnitudeEigenvalue(rows(a))

    /**
     * Spectral condition number of a symmetric matrix `max|λ_i| / min|λ_i|`.
     * Returns [Double.POSITIVE_INFINITY] when `min|λ_i| = 0`.
     */
    public fun conditionSymmetric(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): Double {
        var lo = Double.POSITIVE_INFINITY
        var hi = 0.0
        for (v in symmetricEigenvalues(a, backend)) {
            lo = minOf(lo, abs(v))
            hi = maxOf(hi, abs(v))
        }
        return if (lo == 0.0) Double.POSITIVE_INFINITY else hi / lo
    }

    /** Spectral condition number over a row array; see the [DenseMatrix] overload. */
    public fun conditionSymmetric(a: Array<DoubleArray>): Double = conditionSymmetric(rows(a))

    /**
     * Relative backward error of the solution of `A x = b`:
     * `ω = ‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)`.
     *
     * The normalization matches the post-check of [LinearAlgebra.solve] against the threshold
     * [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE]. The quantity is measured, not estimated,
     * and is therefore always reliable; it does not bound the forward error — the multiplier is
     * `cond(A)`, see [ForwardError]. For zero scale (`A = 0`, `b = 0`) `0.0` is returned.
     *
     * @throws IllegalArgumentException if the matrix is empty, not square, or the lengths of `b`, `x`
     *   do not equal its order.
     */
    public fun relativeBackwardError(
        a: DenseMatrix,
        b: DoubleArray,
        x: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): Double {
        Shapes.requireSquare(a, "matrix A")
        val n = a.rows
        Shapes.requireVectorLength(b, n, "b")
        Shapes.requireVectorLength(x, n, "x")
        val r = backend.matVec(a, x)
        for (i in 0 until n) r[i] -= b[i]
        val scale = maxOf(backend.norm(a, MatrixNorm.INF) * LinearAlgebra.normInf(x), LinearAlgebra.normInf(b))
        return if (scale == 0.0) 0.0 else LinearAlgebra.normInf(r) / scale
    }

    /** Backward error over a row array; see the [DenseMatrix] overload. */
    public fun relativeBackwardError(a: Array<DoubleArray>, b: DoubleArray, x: DoubleArray): Double =
        relativeBackwardError(rows(a), b, x)

    /**
     * Relative forward-error bound `cond·ω` from a ready condition estimate
     * and the measured backward error `ω`.
     *
     * The function computes nothing anew: it converts the pair "`cond` estimate + backward
     * error" into one of the three [ForwardError] modes, never letting an unreliable `cond`
     * turn into a number.
     *
     * @param backwardError the result of [relativeBackwardError]; must be finite and non-negative.
     */
    public fun forwardError(condition: ConditionEstimate, backwardError: Double): ForwardError {
        require(backwardError.isFinite() && backwardError >= 0.0) {
            "forwardError: backward error must be finite and non-negative, got $backwardError"
        }
        return when {
            !condition.condInf.isFinite() -> ForwardError.NoFiniteBound(backwardError)
            !condition.isReliable -> ForwardError.Unreliable(backwardError, condition)
            else -> ForwardError.Bounded(backwardError, condition.condInf, condition.condInf * backwardError)
        }
    }

    /**
     * Relative forward-error bound for a symmetric matrix via the spectral
     * estimate [conditionSymmetric].
     *
     * Distinguishes two facts: there is no finite condition number at all (`min|λ| = 0`,
     * mode [ForwardError.NoFiniteBound]) and `cond` is large but finite
     * ([ForwardError.Bounded] with a large bound). The [ForwardError.Unreliable] mode
     * is never returned.
     *
     * @throws IllegalArgumentException if the matrix is not symmetric.
     */
    public fun forwardErrorSymmetric(
        a: DenseMatrix,
        backwardError: Double,
        backend: LinAlgBackend = Backends.default(),
    ): ForwardError {
        require(backwardError.isFinite() && backwardError >= 0.0) {
            "forwardErrorSymmetric: backward error must be finite and non-negative, got $backwardError"
        }
        val cond = conditionSymmetric(a, backend)
        return if (!cond.isFinite()) {
            ForwardError.NoFiniteBound(backwardError)
        } else {
            ForwardError.Bounded(backwardError, cond, cond * backwardError)
        }
    }

    /** Forward-error bound for a symmetric matrix over a row array; see the [DenseMatrix] overload. */
    public fun forwardErrorSymmetric(a: Array<DoubleArray>, backwardError: Double): ForwardError =
        forwardErrorSymmetric(rows(a), backwardError)

    /** Row-array adapter: an empty array is rejected here, a ragged one in [DenseMatrix.fromRows]. */
    private fun rows(a: Array<DoubleArray>): DenseMatrix {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matrix A must not be empty" }
        return DenseMatrix.fromRows(a)
    }

    // The NumericsContext overloads use its linear-algebra implementation.

    /** Same as [inverse], with the linear-algebra implementation from [context]. */
    public fun inverse(a: DenseMatrix, context: NumericsContext): DenseMatrix? = inverse(a, context.backend)

    /** Same as [conditionInf], with the linear-algebra implementation from [context]. */
    public fun conditionInf(
        a: DenseMatrix,
        context: NumericsContext,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate = conditionInf(a, tolerance, context.backend)

    /** Same as [conditionEstimate], with the linear-algebra implementation from [context]. */
    public fun conditionEstimate(
        a: DenseMatrix,
        context: NumericsContext,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate = conditionEstimate(a, tolerance, context.backend)

    /** Same as [symmetricEigenvalues], with the linear-algebra implementation from [context]. */
    public fun symmetricEigenvalues(a: DenseMatrix, context: NumericsContext): DoubleArray =
        symmetricEigenvalues(a, context.backend)

    /** Same as [conditionSymmetric], with the linear-algebra implementation from [context]. */
    public fun conditionSymmetric(a: DenseMatrix, context: NumericsContext): Double =
        conditionSymmetric(a, context.backend)
}
