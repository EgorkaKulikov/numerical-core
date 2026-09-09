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
 * Независимый оракул на Hipparchus (чистая Java, без BLAS/LAPACK) для property-based тестов.
 * Работает над [DenseMatrix]/[DoubleArray]; конвертация — через строки.
 */
object HipparchusOracle {

    fun toReal(a: DenseMatrix): RealMatrix = MatrixUtils.createRealMatrix(a.toRows())

    fun fromReal(m: RealMatrix): DenseMatrix = DenseMatrix.fromRows(m.data)

    /** Решение A·x = b через LU-разложение Hipparchus (частичный выбор ведущего элемента). */
    fun solve(a: DenseMatrix, b: DoubleArray): DoubleArray =
        LUDecomposition(toReal(a)).solver.solve(ArrayRealVector(b)).toArray()

    /** Обратная матрица через LU-разложение Hipparchus. */
    fun inverse(a: DenseMatrix): DenseMatrix = fromReal(LUDecomposition(toReal(a)).solver.inverse)

    /** Произведение A·x. */
    fun matVec(a: DenseMatrix, x: DoubleArray): DoubleArray = toReal(a).operate(x)

    /** Произведение A·B. */
    fun matMat(a: DenseMatrix, b: DenseMatrix): DenseMatrix = fromReal(toReal(a).multiply(toReal(b)))

    /** Нижний треугольный множитель L разложения Холецкого A = L·Lᵀ. */
    fun cholesky(a: DenseMatrix): DenseMatrix = fromReal(CholeskyDecomposition(toReal(a), 1e-12, 1e-14).l)

    /** Собственные значения симметричной матрицы по возрастанию. */
    fun symmetricEigenvalues(a: DenseMatrix): DoubleArray =
        EigenDecompositionSymmetric(toReal(a)).eigenvalues.sortedArray()

    /** Число обусловленности в норме-1: ‖A‖₁·‖A⁻¹‖₁ (точное, через явное обращение). */
    fun condition1(a: DenseMatrix): Double {
        val m = toReal(a)
        val inv = LUDecomposition(m).solver.inverse
        return m.norm1 * inv.norm1
    }

    /** Узлы и веса квадратуры Гаусса–Лежандра на [-1, 1], узлы по возрастанию. */
    fun gaussLegendre(m: Int): Pair<DoubleArray, DoubleArray> {
        val g = GaussIntegratorFactory().legendre(m)
        val pairs = (0 until g.numberOfPoints).map { g.getPoint(it) to g.getWeight(it) }.sortedBy { it.first }
        return DoubleArray(m) { pairs[it].first } to DoubleArray(m) { pairs[it].second }
    }

    /** Интеграл f по [lo, hi] квадратурой Гаусса–Лежандра с m узлами. */
    fun integrate(lo: Double, hi: Double, m: Int, f: (Double) -> Double): Double =
        GaussIntegratorFactory().legendre(m, lo, hi).integrate(UnivariateFunction { f(it) })
}
