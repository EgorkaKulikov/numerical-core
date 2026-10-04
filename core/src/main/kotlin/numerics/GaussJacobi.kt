package numerics

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Gauss–Jacobi rule with [m] nodes for the weight `(1 − x)^a (1 + x)^b` on `[−1, 1]`, `a, b > −1`:
 * exact for `p·w` with `p` a polynomial of degree at most `2m − 1`.
 *
 * The reference nodes and weights are computed once in the constructor (see [gaussJacobiReference]);
 * [integrate] maps them onto a finite interval and applies the scale factor of the weight analytically.
 *
 * @param m number of nodes (at least 1)
 * @param a exponent of `1 − x`, a finite number greater than −1
 * @param b exponent of `1 + x`, a finite number greater than −1
 */
public class GaussJacobi(public val m: Int, public val a: Double, public val b: Double) {

    private val refNodes: DoubleArray
    private val refWeights: DoubleArray

    init {
        val (nodes, weights) = gaussJacobiReference(m, a, b)
        refNodes = nodes
        refWeights = weights
    }

    /** Reference nodes (ascending) and weights on `[−1, 1]` (copies of the internal arrays are returned). */
    public fun refNodesWeights(): Pair<DoubleArray, DoubleArray> = refNodes.copyOf() to refWeights.copyOf()

    /**
     * Integral `∫_lo^hi (hi − s)^a (s − lo)^b f(s) ds` over a finite interval `lo < hi`.
     *
     * The weight is not evaluated: with `s = mid + half·x`, `mid = (lo + hi)/2`, `half = (hi − lo)/2`,
     * one has `hi − s = half·(1 − x)` and `s − lo = half·(1 + x)`, so the reference rule is applied
     * to `f(mid + half·x)` and the sum is multiplied by `half^(a + b + 1)`.
     *
     * @throws IllegalArgumentException if `lo` or `hi` is not finite or `lo >= hi`
     */
    public fun integrate(lo: Double, hi: Double, f: (Double) -> Double): Double {
        require(lo.isFinite() && hi.isFinite()) { "interval endpoints must be finite, got [$lo, $hi]" }
        require(lo < hi) { "interval endpoints must satisfy lo < hi, got [$lo, $hi]" }
        val mid = 0.5 * (lo + hi)
        val half = 0.5 * (hi - lo)
        var sum = 0.0
        for (i in refNodes.indices) sum += refWeights[i] * f(mid + half * refNodes[i])
        return half.pow(a + b + 1.0) * sum
    }

    /** Computation of the reference Gauss–Jacobi nodes and weights. */
    public companion object {
        private const val BISECTION_STEPS = 40
        private const val MAX_NEWTON_ITERATIONS = 100
        private const val NEWTON_TOLERANCE = 1e-14
        private const val PIVOT_MIN = 1e-300

        /**
         * Gauss–Jacobi nodes (ascending) and weights on `[−1, 1]` for `m` points and the weight
         * `(1 − x)^a (1 + x)^b`; the rule is exact for `p·w` with `p` of degree at most `2m − 1`.
         *
         * The nodes are the eigenvalues of the symmetric tridiagonal Jacobi matrix built from the
         * three-term recurrence `p_{k+1}(x) = (x − α_k)·p_k(x) − β_k·p_{k−1}(x)` of the monic Jacobi
         * polynomials. They are located by Sturm-sequence bisection and refined by Newton's method on
         * `P_m^{(a,b)}`, evaluated through the same recurrence in orthonormal form. The weights are the
         * Christoffel numbers `w_i = μ0 / Σ_{k<m} p̂_k(x_i)²`, where `p̂_k` are the orthonormal
         * polynomials scaled to `p̂_0 = 1` and `μ0 = 2^(a+b+1)·B(a + 1, b + 1)` is the integral of the
         * weight. Only scalar arithmetic is used, so the result does not depend on the BLAS/LAPACK
         * implementation.
         *
         * @throws IllegalArgumentException if `m < 1` or `a`, `b` are not finite numbers greater than −1
         * @throws IllegalStateException if the Newton iterations for some node did not converge
         *   (the exception guards against an infinite loop; in practice it never occurs)
         */
        public fun gaussJacobiReference(m: Int, a: Double, b: Double): Pair<DoubleArray, DoubleArray> =
            computeReference(m, a, b, MAX_NEWTON_ITERATIONS)

        /** [gaussJacobiReference] with an explicit cap on the number of Newton iterations per node. */
        internal fun computeReference(
            m: Int,
            a: Double,
            b: Double,
            maxNewtonIterations: Int,
        ): Pair<DoubleArray, DoubleArray> {
            require(m >= 1) { "number of nodes must be at least 1, got $m" }
            require(a > -1.0 && a.isFinite()) { "exponent a must be a finite number greater than -1, got $a" }
            require(b > -1.0 && b.isFinite()) { "exponent b must be a finite number greater than -1, got $b" }
            val alpha = DoubleArray(m) { k -> recurrenceAlpha(k, a, b) }
            // beta[k] = β_k for k = 1..m; beta[0] = 0 because it multiplies p_{−1} = 0.
            val beta = DoubleArray(m + 1) { k -> if (k == 0) 0.0 else recurrenceBeta(k, a, b) }
            val offDiag = DoubleArray(m + 1) { k -> sqrt(beta[k]) }
            val mu0 = exp(
                (a + b + 1.0) * ln(2.0) + SpecialFunctions.lnGamma(a + 1.0) +
                    SpecialFunctions.lnGamma(b + 1.0) - SpecialFunctions.lnGamma(a + b + 2.0),
            )
            val nodes = DoubleArray(m)
            val weights = DoubleArray(m)
            val values = DoubleArray(3)
            for (i in 0 until m) {
                var x = bisectEigenvalue(i, alpha, beta)
                var converged = false
                for (iter in 0 until maxNewtonIterations) {
                    evaluateOrthonormal(x, alpha, offDiag, values)
                    val dx = values[0] / values[1]
                    x -= dx
                    if (abs(dx) < NEWTON_TOLERANCE) {
                        converged = true
                        break
                    }
                }
                check(converged) { "Newton iterations for node $i did not converge in $maxNewtonIterations steps" }
                evaluateOrthonormal(x, alpha, offDiag, values)
                nodes[i] = x
                weights[i] = mu0 / values[2]
            }
            return nodes to weights
        }

        /** Diagonal coefficient `α_k` of the monic Jacobi recurrence; `k = 0` avoids the 0/0 form at `a + b = 0`. */
        private fun recurrenceAlpha(k: Int, a: Double, b: Double): Double {
            if (k == 0) return (b - a) / (a + b + 2.0)
            val s = 2.0 * k + a + b
            return (b - a) * (b + a) / (s * (s + 2.0))
        }

        /** Coefficient `β_k`, `k ≥ 1`, of the monic Jacobi recurrence; `k = 1` avoids the 0/0 form at `a + b = −1`. */
        private fun recurrenceBeta(k: Int, a: Double, b: Double): Double {
            if (k == 1) return 4.0 * (1.0 + a) * (1.0 + b) / ((2.0 + a + b) * (2.0 + a + b) * (3.0 + a + b))
            val s = 2.0 * k + a + b
            return 4.0 * k * (k + a) * (k + b) * (k + a + b) / (s * s * (s + 1.0) * (s - 1.0))
        }

        /**
         * Eigenvalue number `index` (ascending, from 0) of the Jacobi matrix with diagonal `alpha` and
         * squared off-diagonal `beta[1..]`, located by bisection on `[−1, 1]` to about `2^(1 − BISECTION_STEPS)`.
         */
        private fun bisectEigenvalue(index: Int, alpha: DoubleArray, beta: DoubleArray): Double {
            var lo = -1.0
            var hi = 1.0
            repeat(BISECTION_STEPS) {
                val mid = 0.5 * (lo + hi)
                if (countBelow(mid, alpha, beta) > index) hi = mid else lo = mid
            }
            return 0.5 * (lo + hi)
        }

        /** Number of eigenvalues less than `x`: negative pivots of the LDLᵀ factorization of `T − x·I` (Sturm count). */
        private fun countBelow(x: Double, alpha: DoubleArray, beta: DoubleArray): Int {
            var count = 0
            var d = 1.0
            for (k in alpha.indices) {
                d = alpha[k] - x - beta[k] / d
                if (abs(d) < PIVOT_MIN) d = -PIVOT_MIN
                if (d < 0.0) count++
            }
            return count
        }

        /**
         * Writes `p̂_m(x)`, `p̂_m'(x)` and `Σ_{k<m} p̂_k(x)²` into `out` for the orthonormal recurrence
         * `√β_{k+1}·p̂_{k+1} = (x − α_k)·p̂_k − √β_k·p̂_{k−1}`, `p̂_0 = 1`, `p̂_{−1} = 0`.
         */
        private fun evaluateOrthonormal(x: Double, alpha: DoubleArray, offDiag: DoubleArray, out: DoubleArray) {
            var pPrev = 0.0
            var p = 1.0
            var dPrev = 0.0
            var d = 0.0
            var sumSq = 0.0
            for (k in alpha.indices) {
                sumSq += p * p
                val t = x - alpha[k]
                val pNext = (t * p - offDiag[k] * pPrev) / offDiag[k + 1]
                val dNext = (p + t * d - offDiag[k] * dPrev) / offDiag[k + 1]
                pPrev = p
                p = pNext
                dPrev = d
                d = dNext
            }
            out[0] = p
            out[1] = d
            out[2] = sumSq
        }
    }
}
