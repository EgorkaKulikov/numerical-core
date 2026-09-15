package numerics

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Scalar and utility operations of dense linear algebra over row arrays:
 * matrix constructors, vector norms and the asymmetry measure. They never call
 * into the linear-algebra backend.
 *
 * Internal operations without argument validation; the caller is responsible for
 * checks — [LinearAlgebra], through which these operations are exposed publicly.
 */
internal object DenseOps {

    /** Creates a zero matrix of size rows x cols. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) }

    /** Identity matrix of size n x n. */
    fun identity(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    /** Euclidean norm of a vector. */
    fun norm2(x: DoubleArray): Double = sqrt(x.fold(0.0) { acc, v -> acc + v * v })

    /** Infinity (uniform) norm of a vector. */
    fun normInf(x: DoubleArray): Double = x.fold(0.0) { acc, v -> maxOf(acc, abs(v)) }

    /** Asymmetry measure max|A - A^T| without input validation (performed by [LinearAlgebra.maxAsymmetry]). */
    fun maxAsymmetry(a: Array<DoubleArray>): Double {
        var m = 0.0
        for (i in a.indices) for (j in a.indices) m = maxOf(m, abs(a[i][j] - a[j][i]))
        return m
    }
}
