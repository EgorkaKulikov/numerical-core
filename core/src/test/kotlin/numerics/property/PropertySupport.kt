package numerics.property

import numerics.DenseMatrix
import kotlin.math.abs
import kotlin.math.max

/** Shared helpers for property tests: norms and relative differences computed without calling the library under test. */
internal object PropertySupport {
    fun maxAbs(x: DoubleArray): Double {
        var m = 0.0
        for (v in x) m = max(m, abs(v))
        return m
    }

    /** max|a−b| / max(max|b|, 1). */
    fun relDiff(a: DoubleArray, b: DoubleArray): Double {
        require(a.size == b.size) { "sizes ${a.size} and ${b.size} do not match" }
        var d = 0.0
        for (i in a.indices) d = max(d, abs(a[i] - b[i]))
        return d / max(maxAbs(b), 1.0)
    }

    fun relDiff(a: DenseMatrix, b: DenseMatrix): Double = relDiff(a.data, b.data)

    /** ‖A‖∞ — maximum absolute row sum. */
    fun normInf(a: DenseMatrix): Double {
        var m = 0.0
        for (i in 0 until a.rows) {
            var s = 0.0
            for (j in 0 until a.cols) s += abs(a[i, j])
            m = max(m, s)
        }
        return m
    }

    /** ‖A‖₁ — maximum absolute column sum. */
    fun norm1(a: DenseMatrix): Double = normInf(a.transpose())

    /** Naive A·x (plain loops, independent of any backend). */
    fun plainMatVec(a: DenseMatrix, x: DoubleArray): DoubleArray =
        DoubleArray(a.rows) { i -> var s = 0.0; for (j in 0 until a.cols) s += a[i, j] * x[j]; s }

    /** |A|·|x| — rounding scale for the product. */
    fun absMatVec(a: DenseMatrix, x: DoubleArray): DoubleArray =
        DoubleArray(a.rows) { i -> var s = 0.0; for (j in 0 until a.cols) s += abs(a[i, j] * x[j]); s }

    fun sub(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(a.size) { a[it] - b[it] }

    /** ‖M − I‖∞. */
    fun residualToIdentity(m: DenseMatrix): Double =
        normInf(DenseMatrix.build(m.rows, m.cols) { i, j -> m[i, j] - if (i == j) 1.0 else 0.0 })
}
