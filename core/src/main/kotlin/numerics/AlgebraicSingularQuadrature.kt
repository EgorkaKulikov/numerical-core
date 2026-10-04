package numerics

import kotlin.math.abs
import kotlin.math.pow

/**
 * Composite product quadrature for `∫ |t − s|^(−alpha) f(s) ds` over a partition, `f` smooth on each cell.
 *
 * This is the weakly singular (Abel-type) kernel of Volterra and Fredholm integral operators;
 * `0 ≤ alpha < 1`, and `t` is any finite real number — inside a cell, at a breakpoint, or outside the partition.
 * Each cell `[c0, c1]` is treated as follows.
 * - `t` strictly inside the cell: the cell is split at `t`; the left part is integrated by the [GaussJacobi] rule
 *   with the weight `(t − s)^(−alpha)`, the right part by the one with the weight `(s − t)^(−alpha)`
 *   (`nodesPerSub` nodes each). `t` at an end of the cell: the corresponding rule on the whole cell, no splitting.
 *   The singular factor is absorbed into the weights, so these cells are integrated exactly for polynomial `f`
 *   of degree at most `2·nodesPerSub − 1`.
 * - `t` outside the cell at distance `d` from its nearest end: if `d < c1 − c0`, the cell is refined geometrically
 *   toward that end into pieces at distances `d, 2d, 4d, …` from `t`, each no longer than its distance to `t`;
 *   the refinement stops as soon as the rest of the cell satisfies the same bound and becomes the last piece.
 *   Otherwise the whole cell is one piece. Every piece is integrated by the `nodesPerSub`-point [GaussLegendre]
 *   rule with the kernel evaluated explicitly; the distance `|s − t|` is accumulated from `d` and is free of
 *   cancellation. The number of pieces per cell is capped at [MAX_PIECES]: the cap is reached only when
 *   `d < 2^(−59)·(c1 − c0)`, and then the last piece is longer than its distance to `t` and the accuracy degrades.
 * - `alpha == 0`: the kernel is identically one and the rule is [GaussLegendre.integrate] on the same partition.
 *
 * Grading of the partition toward the ends of the integration interval (to resolve endpoint singularities of `f`)
 * is not supported: the caller supplies the partition.
 *
 * @param alpha exponent of the kernel, `0 ≤ alpha < 1`
 * @param nodesPerSub number of nodes of every Gauss–Jacobi and Gauss–Legendre rule, at least 1
 * @throws IllegalArgumentException if `alpha` is outside `[0, 1)` or `nodesPerSub < 1`
 */
public class AlgebraicSingularQuadrature(public val alpha: Double, public val nodesPerSub: Int = 12) {
    init {
        require(alpha >= 0.0 && alpha < 1.0) { "kernel exponent alpha must lie in [0, 1), got $alpha" }
        require(nodesPerSub >= 1) { "number of nodes per subinterval must be at least 1, got $nodesPerSub" }
    }

    private val legendre = GaussLegendre(nodesPerSub)
    private val legendreNodes: DoubleArray
    private val legendreWeights: DoubleArray

    /** Weight `(hi − s)^(−alpha)` on `[lo, hi]`: the part of a cell to the left of `t`. */
    private val leftOfT = GaussJacobi(nodesPerSub, -alpha, 0.0)

    /** Weight `(s − lo)^(−alpha)` on `[lo, hi]`: the part of a cell to the right of `t`. */
    private val rightOfT = GaussJacobi(nodesPerSub, 0.0, -alpha)

    init {
        val (nodes, weights) = legendre.refNodesWeights()
        legendreNodes = nodes
        legendreWeights = weights
    }

    /**
     * Integral `∫_{bp.first}^{bp.last} |t − s|^(−alpha) f(s) ds` over the partition `breakpoints`: the points are
     * strictly increasing, the first and the last are the ends of the integration interval. `f` is sampled only
     * at quadrature nodes inside the cells. For the Volterra operator `∫_0^t` pass the partition truncated at `t`.
     *
     * @throws IllegalArgumentException if there are fewer than two breakpoints, the breakpoints are not finite and
     *   strictly increasing, or `t` is not finite
     */
    public fun integrate(breakpoints: DoubleArray, t: Double, f: (Double) -> Double): Double {
        require(t.isFinite()) { "evaluation point t must be finite, got $t" }
        require(breakpoints.size >= 2) { "partition must contain at least two points, got ${breakpoints.size}" }
        require(breakpoints.first().isFinite() && breakpoints.last().isFinite()) {
            "partition endpoints must be finite, got [${breakpoints.first()}, ${breakpoints.last()}]"
        }
        for (k in 0 until breakpoints.size - 1) {
            require(breakpoints[k] < breakpoints[k + 1]) {
                "breakpoints must be strictly increasing; violated between positions $k and ${k + 1}"
            }
        }
        if (alpha == 0.0) return legendre.integrate(breakpoints, f)
        var sum = 0.0
        for (k in 0 until breakpoints.size - 1) {
            sum += integrateCell(breakpoints[k], breakpoints[k + 1], t, f)
        }
        return sum
    }

    private fun integrateCell(c0: Double, c1: Double, t: Double, f: (Double) -> Double): Double =
        when {
            t < c0 -> integrateAway(c0, c1, c0 - t, 1.0, f)
            t > c1 -> integrateAway(c1, c0, t - c1, -1.0, f)
            t == c0 -> rightOfT.integrate(c0, c1, f)
            t == c1 -> leftOfT.integrate(c0, c1, f)
            else -> leftOfT.integrate(c0, t, f) + rightOfT.integrate(t, c1, f)
        }

    /**
     * Cell on one side of `t`: `near` is its end closest to `t`, at distance `d > 0`; `dir` is the direction from
     * `near` into the cell. Pieces `[r, 2r]` in the distance `r` from `t`, starting at `r = d`, until the rest of the
     * cell is no longer than its distance to `t` (or [MAX_PIECES] is reached).
     */
    private fun integrateAway(near: Double, far: Double, d: Double, dir: Double, f: (Double) -> Double): Double {
        val length = abs(far - near)
        var r = d
        var pieces = 1
        var sum = 0.0
        while (pieces < MAX_PIECES && length - (r - d) > r) {
            sum += legendrePiece(near + dir * (r - d), r, r, dir, f)
            r *= 2.0
            pieces++
        }
        return sum + legendrePiece(near + dir * (r - d), r, length - (r - d), dir, f)
    }

    /**
     * Gauss–Legendre on the piece of length `h` that starts at `start` (at distance `u0` from `t`) and extends in the
     * direction `dir` away from `t`; the kernel is `(u0 + offset)^(−alpha)`.
     */
    private fun legendrePiece(start: Double, u0: Double, h: Double, dir: Double, f: (Double) -> Double): Double {
        val half = 0.5 * h
        var sum = 0.0
        for (q in legendreNodes.indices) {
            val offset = half * (1.0 + legendreNodes[q])
            sum += legendreWeights[q] * (u0 + offset).pow(-alpha) * f(start + dir * offset)
        }
        return half * sum
    }

    public companion object {
        /**
         * Maximum number of Gauss–Legendre pieces per cell lying on one side of `t` (geometric refinement toward
         * `t` included); it bounds the work when `t` almost touches a cell.
         */
        public const val MAX_PIECES: Int = 60
    }
}
