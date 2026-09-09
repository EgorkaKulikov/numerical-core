package numerics.property

import numerics.DenseMatrix
import kotlin.math.abs
import kotlin.math.max

/** Общие хелперы property-тестов: нормы и относительные расхождения без обращения к проверяемой библиотеке. */
internal object PropertySupport {
    fun maxAbs(x: DoubleArray): Double {
        var m = 0.0
        for (v in x) m = max(m, abs(v))
        return m
    }

    /** max|a−b| / max(max|b|, 1). */
    fun relDiff(a: DoubleArray, b: DoubleArray): Double {
        require(a.size == b.size) { "размеры ${a.size} и ${b.size} не совпадают" }
        var d = 0.0
        for (i in a.indices) d = max(d, abs(a[i] - b[i]))
        return d / max(maxAbs(b), 1.0)
    }

    fun relDiff(a: DenseMatrix, b: DenseMatrix): Double = relDiff(a.data, b.data)

    /** ‖A‖∞ — максимальная сумма модулей по строкам. */
    fun normInf(a: DenseMatrix): Double {
        var m = 0.0
        for (i in 0 until a.rows) {
            var s = 0.0
            for (j in 0 until a.cols) s += abs(a[i, j])
            m = max(m, s)
        }
        return m
    }

    /** ‖A‖₁ — максимальная сумма модулей по столбцам. */
    fun norm1(a: DenseMatrix): Double = normInf(a.transpose())

    /** Наивное A·x (тройной цикл, независимо от бэкендов). */
    fun plainMatVec(a: DenseMatrix, x: DoubleArray): DoubleArray =
        DoubleArray(a.rows) { i -> var s = 0.0; for (j in 0 until a.cols) s += a[i, j] * x[j]; s }

    /** |A|·|x| — масштаб округления для произведения. */
    fun absMatVec(a: DenseMatrix, x: DoubleArray): DoubleArray =
        DoubleArray(a.rows) { i -> var s = 0.0; for (j in 0 until a.cols) s += abs(a[i, j] * x[j]); s }

    fun sub(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(a.size) { a[it] - b[it] }

    /** ‖M − I‖∞. */
    fun residualToIdentity(m: DenseMatrix): Double =
        normInf(DenseMatrix.build(m.rows, m.cols) { i, j -> m[i, j] - if (i == j) 1.0 else 0.0 })
}
