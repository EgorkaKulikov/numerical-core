package numerics.oracle

import numerics.DenseMatrix
import org.hipparchus.analysis.UnivariateFunction
import org.hipparchus.analysis.integration.gauss.GaussIntegratorFactory
import org.hipparchus.linear.ArrayRealVector
import org.hipparchus.linear.CholeskyDecomposition
import org.hipparchus.linear.EigenDecompositionSymmetric
import org.hipparchus.linear.LUDecomposition
import org.hipparchus.linear.MatrixUtils
import org.hipparchus.linear.RealMatrix

/**
 * Independent oracle backed by Hipparchus (pure Java, no BLAS/LAPACK) for property-based tests.
 * Operates on [DenseMatrix]/[DoubleArray]; conversion goes through row arrays.
 */
object HipparchusOracle {

    fun toReal(a: DenseMatrix): RealMatrix = MatrixUtils.createRealMatrix(a.toRows())

    fun fromReal(m: RealMatrix): DenseMatrix = DenseMatrix.fromRows(m.data)

    /** Solves A·x = b via the Hipparchus LU factorization (partial pivoting). */
    fun solve(a: DenseMatrix, b: DoubleArray): DoubleArray =
        LUDecomposition(toReal(a)).solver.solve(ArrayRealVector(b)).toArray()

    /** Inverse matrix via the Hipparchus LU factorization. */
    fun inverse(a: DenseMatrix): DenseMatrix = fromReal(LUDecomposition(toReal(a)).solver.inverse)

    /** Product A·x. */
    fun matVec(a: DenseMatrix, x: DoubleArray): DoubleArray = toReal(a).operate(x)

    /** Product A·B. */
    fun matMat(a: DenseMatrix, b: DenseMatrix): DenseMatrix = fromReal(toReal(a).multiply(toReal(b)))

    /** Lower triangular factor L of the Cholesky factorization A = L·Lᵀ. */
    fun cholesky(a: DenseMatrix): DenseMatrix = fromReal(CholeskyDecomposition(toReal(a), 1e-12, 1e-14).l)

    /** Eigenvalues of a symmetric matrix in ascending order. */
    fun symmetricEigenvalues(a: DenseMatrix): DoubleArray =
        EigenDecompositionSymmetric(toReal(a)).eigenvalues.sortedArray()

    /** Condition number in the 1-norm: ‖A‖₁·‖A⁻¹‖₁ (exact, via explicit inversion). */
    fun condition1(a: DenseMatrix): Double {
        val m = toReal(a)
        val inv = LUDecomposition(m).solver.inverse
        return m.norm1 * inv.norm1
    }

    /** Gauss–Legendre nodes and weights on [-1, 1], nodes in ascending order. */
    fun gaussLegendre(m: Int): Pair<DoubleArray, DoubleArray> {
        val g = GaussIntegratorFactory().legendre(m)
        val pairs = (0 until g.numberOfPoints).map { g.getPoint(it) to g.getWeight(it) }.sortedBy { it.first }
        return DoubleArray(m) { pairs[it].first } to DoubleArray(m) { pairs[it].second }
    }

    /** Integral of f over [lo, hi] by Gauss–Legendre quadrature with m nodes. */
    fun integrate(lo: Double, hi: Double, m: Int, f: (Double) -> Double): Double =
        GaussIntegratorFactory().legendre(m, lo, hi).integrate(UnivariateFunction { f(it) })
}
