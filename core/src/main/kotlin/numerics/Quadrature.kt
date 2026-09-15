package numerics

import kotlin.math.abs

/**
 * Composite Gauss–Legendre quadrature: on each subinterval of the partition a rule with
 * `nodesPerSub` nodes is applied, exact for polynomials of degree `2·nodesPerSub − 1`.
 *
 * The reference nodes and weights on `[-1, 1]` are computed once in the constructor
 * (see [gaussLegendreReference]) and mapped onto each subinterval by a linear change of variable.
 *
 * @param nodesPerSub number of nodes on each subinterval (at least 1)
 */
public class GaussLegendre(public val nodesPerSub: Int = 8) {

    private val refNodes: DoubleArray
    private val refWeights: DoubleArray

    init {
        require(nodesPerSub >= 1) { "number of nodes per subinterval must be at least 1, got $nodesPerSub" }
        val (nodes, weights) = gaussLegendreReference(nodesPerSub)
        refNodes = nodes
        refWeights = weights
    }

    /**
     * Integral of `f` over the composite partition `breakpoints`: the points are strictly increasing,
     * the first and the last are the endpoints of the integration interval.
     *
     * @throws IllegalArgumentException if there are fewer than two points or they are not strictly increasing
     */
    public fun integrate(breakpoints: DoubleArray, f: (Double) -> Double): Double {
        require(breakpoints.size >= 2) { "partition must contain at least two points, got ${breakpoints.size}" }
        for (k in 0 until breakpoints.size - 1) {
            require(breakpoints[k] < breakpoints[k + 1]) {
                "breakpoints must be strictly increasing; violated between positions $k and ${k + 1}"
            }
        }
        var sum = 0.0
        for (k in 0 until breakpoints.size - 1) {
            val lo = breakpoints[k]
            val hi = breakpoints[k + 1]
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in refNodes.indices) {
                val t = mid + half * refNodes[q]
                sum += half * refWeights[q] * f(t)
            }
        }
        return sum
    }

    /**
     * Integral over a single interval from `lo` to `hi`. `lo > hi` is allowed — the endpoints
     * are swapped and the result is negated; for `lo == hi` 0 is returned.
     */
    public fun integrateInterval(lo: Double, hi: Double, f: (Double) -> Double): Double =
        when {
            lo == hi -> 0.0
            lo > hi -> -integrateInterval(hi, lo, f)
            else -> integrate(doubleArrayOf(lo, hi), f)
        }

    /** Reference nodes and weights on `[-1, 1]` (copies of the internal arrays are returned). */
    public fun refNodesWeights(): Pair<DoubleArray, DoubleArray> = refNodes.copyOf() to refWeights.copyOf()

    /** Computation of the reference Gauss–Legendre nodes and weights. */
    public companion object {
        /**
         * Gauss–Legendre nodes and weights on `[-1, 1]` for `m` points: Newton's method on the zeros
         * of the Legendre polynomial `P_m` from the initial guess `cos(π(i + 3/4)/(m + 1/2))`,
         * weights `2 / ((1 − x²)·P_m'(x)²)`. The rule is exact for polynomials of degree `2m − 1`.
         *
         * @throws IllegalArgumentException if `m < 1`
         * @throws IllegalStateException if the Newton iterations for some node did not converge
         *   (the exception guards against an infinite loop; in practice it never occurs)
         */
        public fun gaussLegendreReference(m: Int): Pair<DoubleArray, DoubleArray> {
            require(m >= 1) { "number of nodes must be at least 1, got $m" }
            val maxIter = 100
            val nodes = DoubleArray(m)
            val weights = DoubleArray(m)
            for (i in 0 until (m + 1) / 2) {
                var x = Math.cos(Math.PI * (i + 0.75) / (m + 0.5))
                var dp: Double
                var converged = false
                for (iter in 0 until maxIter) {
                    var p0 = 1.0
                    var p1 = x
                    for (k in 2..m) {
                        val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                        p0 = p1; p1 = p2
                    }
                    dp = m * (x * p1 - p0) / (x * x - 1.0)
                    val dx = p1 / dp
                    x -= dx
                    if (abs(dx) < 1e-15) {
                        converged = true
                        break
                    }
                }
                check(converged) { "Newton iterations for node $i did not converge in $maxIter steps" }
                var p0 = 1.0
                var p1 = x
                for (k in 2..m) {
                    val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                    p0 = p1; p1 = p2
                }
                dp = m * (x * p1 - p0) / (x * x - 1.0)
                val w = 2.0 / ((1.0 - x * x) * dp * dp)
                nodes[i] = -x
                nodes[m - 1 - i] = x
                weights[i] = w
                weights[m - 1 - i] = w
            }
            return nodes to weights
        }
    }
}
