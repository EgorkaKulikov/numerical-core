package numerics

import kotlin.math.abs

/**
 * Shared shape validation of inputs for [LinearAlgebra] and [Conditioning].
 * Every check fails with an [IllegalArgumentException] whose message
 * names the actual dimensions.
 */
internal object Shapes {

    /** The matrix must have at least one row and one column. */
    fun requireNonEmpty(a: DenseMatrix, what: String = "matrix") {
        require(a.rows > 0 && a.cols > 0) { "$what must not be empty, got ${a.rows}×${a.cols}" }
    }

    /** The matrix must be non-empty and square. */
    fun requireSquare(a: DenseMatrix, what: String = "matrix") {
        requireNonEmpty(a, what)
        require(a.isSquare) { "$what must be square, got ${a.rows}×${a.cols}" }
    }

    /** The vector length must equal [n]. */
    fun requireVectorLength(v: DoubleArray, n: Int, what: String) {
        require(v.size == n) { "length of vector $what must be $n, got ${v.size}" }
    }

    /** Both matrices are non-empty and the column count of the left one equals the row count of the right one. */
    fun requireMultiplicable(a: DenseMatrix, b: DenseMatrix) {
        requireNonEmpty(a, "left matrix")
        requireNonEmpty(b, "right matrix")
        require(a.cols == b.rows) {
            "dimensions are incompatible for multiplication: ${a.rows}×${a.cols} and ${b.rows}×${b.cols}"
        }
    }

    /** The two matrices have the same shape. */
    fun requireSameShape(a: DenseMatrix, b: DenseMatrix) {
        require(a.rows == b.rows && a.cols == b.cols) {
            "matrix dimensions must match, got ${a.rows}×${a.cols} and ${b.rows}×${b.cols}"
        }
    }

    /** All matrix entries are finite numbers. */
    fun requireFinite(a: DenseMatrix, what: String) {
        require(a.data.all { it.isFinite() }) { "$what contains non-finite values (NaN or infinity)" }
    }

    /** All vector components are finite numbers. */
    fun requireFinite(v: DoubleArray, what: String) {
        require(v.all { it.isFinite() }) { "$what contains non-finite values (NaN or infinity)" }
    }

    /**
     * The square matrix is symmetric up to rounding:
     * `max|A − Aᵀ| ≤ 1e-12 · ‖A‖∞`; the norm [normInf] is supplied by the caller.
     */
    fun requireSymmetric(a: DenseMatrix, normInf: Double, what: String = "matrix") {
        requireSquare(a, what)
        val n = a.rows
        val d = a.data
        var asym = 0.0
        for (j in 0 until n) for (i in j + 1 until n) {
            asym = maxOf(asym, abs(d[i + j * n] - d[j + i * n]))
        }
        require(asym <= 1e-12 * maxOf(normInf, Double.MIN_VALUE)) {
            "$what must be symmetric: max|A−Aᵀ| = $asym"
        }
    }
}
