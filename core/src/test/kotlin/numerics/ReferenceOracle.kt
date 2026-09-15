package numerics

import kotlin.math.abs

/**
 * Independent pure-Kotlin linear algebra implementation for cross-checking the backends
 * in tests: products by plain loops and LU with partial pivoting. Does not reference
 * the [LinearAlgebra] entry point or [numerics.backend.Backends], otherwise the
 * cross-check would be circular.
 */
object ReferenceOracle {

    /** Creates a zero matrix of size rows x cols. */
    private fun zeros(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) }

    /** Product of matrix A (m x k) and vector x (k) -> vector (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray {
        val m = a.size
        val out = DoubleArray(m)
        for (i in 0 until m) {
            var s = 0.0
            val row = a[i]
            for (j in x.indices) s += row[j] * x[j]
            out[i] = s
        }
        return out
    }

    /** Transposed product A^T y, A: m x n, y: m -> vector n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray {
        val m = a.size
        val n = a[0].size
        val out = DoubleArray(n)
        for (i in 0 until m) {
            val row = a[i]
            val yi = y[i]
            for (j in 0 until n) out[j] += row[j] * yi
        }
        return out
    }

    /** Matrix product A (m x k) times B (k x p) -> (m x p). */
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val m = a.size
        val k = b.size
        val p = b[0].size
        val out = zeros(m, p)
        for (i in 0 until m) {
            val ai = a[i]
            val oi = out[i]
            for (l in 0 until k) {
                val ail = ai[l]
                if (ail == 0.0) continue
                val bl = b[l]
                for (j in 0 until p) oi[j] += ail * bl[j]
            }
        }
        return out
    }

    /** Product A^T diag(w) A for A: m x n, w: m -> symmetric n x n. */
    fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> {
        val m = a.size
        val n = a[0].size
        val out = zeros(n, n)
        for (k in 0 until m) {
            val row = a[k]
            val wk = w[k]
            for (i in 0 until n) {
                val rwi = wk * row[i]
                if (rwi == 0.0) continue
                val outI = out[i]
                for (j in 0 until n) outI[j] += rwi * row[j]
            }
        }
        return out
    }

    /** Element-wise sum A + s*B (same dimensions). */
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> {
        val out = Array(a.size) { a[it].copyOf() }
        for (i in a.indices) for (j in a[i].indices) out[i][j] += s * b[i][j]
        return out
    }

    /**
     * Pivot singularity threshold, relative to the matrix scale.
     *
     * Used as `PIVOT_RELATIVE_TOLERANCE * ||A||_inf`. An absolute threshold
     * (formerly 1e-300) effectively checked only for an exact machine zero and let
     * practically singular matrices through: a system with condition number ~1e18
     * was solved without an exception and returned a non-finite result. The value 1e-14 is close to
     * the double machine epsilon (2.2e-16) with a margin for error accumulation in Gaussian elimination.
     */
    private const val PIVOT_RELATIVE_TOLERANCE = 1e-14

    /**
     * Solves the dense linear system A x = b by LU with partial pivoting;
     * A and b are left unchanged. A pivot that is small relative to the matrix norm and
     * a non-finite result are both treated as singularity.
     * @throws IllegalStateException on singularity or a non-finite result.
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val n = a.size
        val lu = Array(n) { a[it].copyOf() }
        val x = b.copyOf()
        // Matrix scale: maximum absolute row sum (the ||A||_inf norm).
        var matrixNorm = 0.0
        for (row in a) {
            var rowSum = 0.0
            for (v in row) rowSum += abs(v)
            matrixNorm = maxOf(matrixNorm, rowSum)
        }
        val pivotTolerance = PIVOT_RELATIVE_TOLERANCE * maxOf(matrixNorm, java.lang.Double.MIN_NORMAL)
        for (col in 0 until n) {
            var pivRow = col
            var pivVal = abs(lu[col][col])
            for (r in col + 1 until n) {
                val v = abs(lu[r][col])
                if (v > pivVal) { pivVal = v; pivRow = r }
            }
            if (pivVal <= pivotTolerance) error("LU: matrix is singular (col=$col)")
            if (pivRow != col) {
                val t = lu[col]; lu[col] = lu[pivRow]; lu[pivRow] = t
                val tx = x[col]; x[col] = x[pivRow]; x[pivRow] = tx
            }
            val pivotR = lu[col]
            val pivot = pivotR[col]
            for (r in col + 1 until n) {
                val factor = lu[r][col] / pivot
                lu[r][col] = factor
                val rowR = lu[r]
                for (c in col + 1 until n) rowR[c] -= factor * pivotR[c]
                x[r] -= factor * x[col]
            }
        }
        // back substitution
        for (i in n - 1 downTo 0) {
            var s = x[i]
            val row = lu[i]
            for (j in i + 1 until n) s -= row[j] * x[j]
            x[i] = s / row[i]
        }
        // Non-finite result handling: if the input contained NaN/Inf, returning a NaN vector
        // without an exception is not acceptable. The same requirement is duplicated in the entry point (for
        // all backends), but it is also needed here when tests call the oracle directly.
        for (v in x) {
            if (v.isNaN() || v.isInfinite()) error("LU: matrix is singular (non-finite result)")
        }
        return x
    }
}

