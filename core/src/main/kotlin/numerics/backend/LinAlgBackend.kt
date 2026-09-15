package numerics.backend

import numerics.DenseMatrix

/**
 * Dense linear algebra backend over [DenseMatrix] matrices (flat column-major storage)
 * and flat [DoubleArray] vectors.
 *
 * This is the pluggable-implementation interface: through it the single entry point
 * [numerics.LinearAlgebra] calls into the computational kernel; the implementation is
 * selected in [Backends] and passed as an explicit parameter.
 * The column-major layout matches the BLAS/LAPACK convention, so data is handed to
 * the kernel without repacking.
 *
 * Implementation contract:
 *  - input matrices and vectors are never modified; where the kernel overwrites its input
 *    with the result, the implementation works on a copy;
 *  - dimension consistency is checked by the caller ([numerics.LinearAlgebra]); the
 *    implementation may rely on it and need not duplicate the checks;
 *  - the result is always a new object that does not share memory with the input.
 */
public interface LinAlgBackend {

    /** Human-readable backend name for logging and diagnostics. */
    public val name: String

    /** True if computations are performed by the system (native) BLAS/LAPACK library. */
    public val isNative: Boolean

    /**
     * In-place vector update `y += alpha·x` (daxpy).
     * @throws IllegalArgumentException if the lengths of `x` and `y` differ.
     */
    public fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray)

    /** Product A·x for A of size m×k and x of length k; the result has length m. */
    public fun matVec(a: DenseMatrix, x: DoubleArray): DoubleArray

    /** Product Aᵀ·y for A of size m×n and y of length m; the result has length n. */
    public fun matTransVec(a: DenseMatrix, y: DoubleArray): DoubleArray

    /** Product A·B for A of size m×k and B of size k×p; the result is m×p. */
    public fun matMat(a: DenseMatrix, b: DenseMatrix): DenseMatrix

    /** Product Aᵀ·B for A of size m×n and B of size m×p; the result is n×p. */
    public fun matTransMat(a: DenseMatrix, b: DenseMatrix): DenseMatrix

    /**
     * Solves the system A·X = B for square A with the columns of B as right-hand sides.
     * LU factorization with partial pivoting.
     *
     * @return matrix X of the same size as B.
     * @throws IllegalStateException if a zero pivot was encountered during the
     *   factorization (the matrix is singular).
     */
    public fun solve(a: DenseMatrix, b: DenseMatrix): DenseMatrix

    /**
     * LU factorization with partial pivoting: P·A = L·U.
     * Detected singularity does not abort the factorization and is reported via
     * [LuFactorization.singularAt].
     */
    public fun luFactor(a: DenseMatrix): LuFactorization

    /**
     * Solves A·X = B using an existing factorization [lu] of the square matrix A.
     * @throws IllegalArgumentException if the factorization is singular ([LuFactorization.isSingular]).
     */
    public fun luSolve(lu: LuFactorization, b: DenseMatrix): DenseMatrix

    /**
     * Inverse matrix as the solution of A·X = I via LU factorization; the residual ‖A·X − I‖
     * is controlled the same way as when solving a system. Returns `null` if A is singular.
     */
    public fun inverse(a: DenseMatrix): DenseMatrix?

    /**
     * Cholesky factorization A = L·Lᵀ for symmetric positive definite A;
     * only the lower triangle of A is read.
     * @return lower-triangular L (entries above the diagonal are zero), or `null`
     *   if A is not positive definite.
     */
    public fun cholesky(a: DenseMatrix): DenseMatrix?

    /**
     * Eigenvalues of a symmetric matrix in ascending order; only the upper
     * triangle of A is read.
     * @throws IllegalStateException if the iterative algorithm failed to converge.
     */
    public fun symmetricEigenvalues(a: DenseMatrix): DoubleArray

    /**
     * Estimate of the reciprocal condition number in the 1-norm: an approximation of
     * `1 / (‖A‖₁ · ‖A⁻¹‖₁)` from the factorization [lu] and the known norm [norm1] of the original matrix.
     * A value of 0 means the matrix is numerically singular.
     */
    public fun reciprocalCondition1(lu: LuFactorization, norm1: Double): Double

    /** Matrix norm of the given kind; 0 for an empty matrix. */
    public fun norm(a: DenseMatrix, kind: MatrixNorm): Double
}

/** Kinds of matrix norms available in [LinAlgBackend.norm]. */
public enum class MatrixNorm {
    /** Maximum absolute column sum, ‖A‖₁. */
    ONE,

    /** Maximum absolute row sum, ‖A‖∞. */
    INF,

    /** Square root of the sum of squares of all entries, ‖A‖F. */
    FROBENIUS,

    /** Largest absolute entry, max|aᵢⱼ| (not a matrix norm in the strict sense). */
    MAX,
}

/**
 * Result of an LU factorization with partial pivoting.
 *
 * @property lu the factors L and U packed in one matrix: unit lower-triangular L below
 *   the diagonal, upper-triangular U on and above the diagonal.
 * @property ipiv row permutations: row i was swapped with row `ipiv[i]` (1-based).
 * @property singularAt 0 if the matrix is non-singular; otherwise the index k (1-based) of the
 *   first zero diagonal entry of U — the factorization is complete, but it cannot be used to solve a system.
 */
public class LuFactorization(public val lu: DenseMatrix, public val ipiv: IntArray, public val singularAt: Int) {
    /** True if a zero pivot was encountered during the factorization. */
    public val isSingular: Boolean get() = singularAt > 0
}
