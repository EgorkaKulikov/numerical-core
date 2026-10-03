package numerics

import org.hipparchus.special.Beta
import org.hipparchus.special.Gamma
import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@Tag("fast")
class GaussJacobiTest {
    private val exponentPairs = listOf(
        -0.5 to 0.0, 0.0 to -0.5, -1.0 / 3 to 0.0, -0.9 to 0.0,
        -0.5 to 1.0, -2.0 / 3 to 2.0, 0.5 to 0.5, -0.5 to -0.5,
    )
    private val nodeCounts = listOf(1, 2, 5, 12, 20)

    /** `∫_{−1}^{1} (1 − x)^a (1 + x)^(b + k) dx = 2^(a+b+k+1)·B(a + 1, b + k + 1)` from the independent oracle. */
    private fun shiftedMoment(a: Double, b: Double, k: Int): Double =
        exp((a + b + k + 1) * ln(2.0) + Beta.logBeta(a + 1, b + k + 1))

    private fun relErr(got: Double, ref: Double): Double = abs(got - ref) / abs(ref)

    @Test fun legendreCaseMatchesGaussLegendre() {
        for (m in 1..40) {
            val (x, w) = GaussJacobi(m, 0.0, 0.0).refNodesWeights()
            val (xl, wl) = GaussLegendre.gaussLegendreReference(m)
            for (i in 0 until m) {
                assertEquals(xl[i], x[i], 1e-14, "node $i, m = $m")
                assertEquals(wl[i], w[i], 1e-14, "weight $i, m = $m")
            }
        }
    }

    @Test fun singleNodeIsAtRatioOfExponents() {
        for ((a, b) in exponentPairs) {
            val (x, w) = GaussJacobi(1, a, b).refNodesWeights()
            assertEquals((b - a) / (a + b + 2), x[0], 1e-15, "a = $a, b = $b")
            assertTrue(relErr(w[0], shiftedMoment(a, b, 0)) <= 1e-14, "a = $a, b = $b")
        }
    }

    @Test fun exactForShiftedPowersUpToDegree2mMinus1() {
        for ((a, b) in exponentPairs) {
            for (m in nodeCounts) {
                val (x, w) = GaussJacobi(m, a, b).refNodesWeights()
                for (k in 0..2 * m - 1) {
                    val got = x.indices.sumOf { i -> w[i] * (1 + x[i]).pow(k) }
                    val err = relErr(got, shiftedMoment(a, b, k))
                    assertTrue(err <= 1e-13, "a = $a, b = $b, m = $m, k = $k: relative error $err")
                }
            }
        }
    }

    @Test fun integratesWeaklySingularMomentsOnInterval() {
        for (alpha in doubleArrayOf(0.1, 1.0 / 3, 0.5, 2.0 / 3, 0.9)) {
            for (m in intArrayOf(3, 8)) {
                val rule = GaussJacobi(m, -alpha, 0.0)
                for (t in doubleArrayOf(0.3, 1.0, 7.0)) {
                    for (j in 0..5) {
                        val got = rule.integrate(0.0, t) { s -> s.pow(j) }
                        val ref = exp((j + 1 - alpha) * ln(t) + Beta.logBeta(1 - alpha, j + 1.0))
                        val err = relErr(got, ref)
                        assertTrue(err <= 1e-13, "alpha = $alpha, m = $m, t = $t, j = $j: relative error $err")
                    }
                }
            }
        }
    }

    @Test fun integrateAppliesBothEndpointFactors() {
        // ∫_1^3 (3 − s)^a (s − 1)^b ds = 2^(a+b+1)·B(a + 1, b + 1).
        val (a, b) = -2.0 / 3 to 2.0
        val got = GaussJacobi(4, a, b).integrate(1.0, 3.0) { 1.0 }
        assertTrue(relErr(got, shiftedMoment(a, b, 0)) <= 1e-14)
    }

    @Test fun nodesIncreaseInsideIntervalAndWeightsSumToMu0() {
        for ((a, b) in exponentPairs) {
            for (m in nodeCounts + 40) {
                val (x, w) = GaussJacobi(m, a, b).refNodesWeights()
                assertTrue(x.first() > -1.0 && x.last() < 1.0, "a = $a, b = $b, m = $m")
                for (i in 1 until m) assertTrue(x[i] > x[i - 1], "a = $a, b = $b, m = $m, i = $i")
                assertTrue(w.all { it > 0.0 }, "a = $a, b = $b, m = $m")
                val err = relErr(w.sum(), shiftedMoment(a, b, 0))
                assertTrue(err <= 1e-14, "a = $a, b = $b, m = $m: relative error $err")
            }
        }
    }

    @Test fun swappingExponentsMirrorsNodes() {
        for ((a, b) in exponentPairs) {
            for (m in nodeCounts) {
                val (x, w) = GaussJacobi(m, a, b).refNodesWeights()
                val (xs, ws) = GaussJacobi(m, b, a).refNodesWeights()
                for (i in 0 until m) {
                    assertEquals(-x[m - 1 - i], xs[i], 1e-15, "a = $a, b = $b, m = $m, i = $i")
                    assertTrue(relErr(ws[i], w[m - 1 - i]) <= 1e-13, "a = $a, b = $b, m = $m, i = $i")
                }
            }
        }
    }

    @Test fun refNodesWeightsReturnsCopies() {
        val rule = GaussJacobi(3, -0.5, 0.0)
        val (x, w) = rule.refNodesWeights()
        x.fill(0.0)
        w.fill(0.0)
        val (x2, w2) = rule.refNodesWeights()
        assertContentEquals(GaussJacobi.gaussJacobiReference(3, -0.5, 0.0).first, x2)
        assertContentEquals(GaussJacobi.gaussJacobiReference(3, -0.5, 0.0).second, w2)
        assertEquals(3, rule.m)
        assertEquals(-0.5, rule.a)
        assertEquals(0.0, rule.b)
    }

    @Test fun rejectsInvalidArguments() {
        assertFailsWith<IllegalArgumentException> { GaussJacobi(0, 0.0, 0.0) }
        assertFailsWith<IllegalArgumentException> { GaussJacobi(3, -1.0, 0.0) }
        assertFailsWith<IllegalArgumentException> { GaussJacobi(3, 0.0, -1.5) }
        assertFailsWith<IllegalArgumentException> { GaussJacobi(3, Double.NaN, 0.0) }
        assertFailsWith<IllegalArgumentException> { GaussJacobi(3, 0.0, Double.NaN) }
        assertFailsWith<IllegalArgumentException> { GaussJacobi(3, Double.POSITIVE_INFINITY, 0.0) }
        assertFailsWith<IllegalArgumentException> { GaussJacobi(3, 0.0, Double.POSITIVE_INFINITY) }
        val rule = GaussJacobi(3, -0.5, 0.0)
        assertFailsWith<IllegalArgumentException> { rule.integrate(1.0, 1.0) { it } }
        assertFailsWith<IllegalArgumentException> { rule.integrate(2.0, 1.0) { it } }
        assertFailsWith<IllegalArgumentException> { rule.integrate(Double.NEGATIVE_INFINITY, 1.0) { it } }
        assertFailsWith<IllegalArgumentException> { rule.integrate(0.0, Double.POSITIVE_INFINITY) { it } }
        assertFailsWith<IllegalArgumentException> { rule.integrate(Double.NaN, 1.0) { it } }
    }

    @Test fun newtonGuardReportsNonConvergence() {
        // One Newton step from the bisection estimate (about 1e-12 away) cannot meet the 1e-14 step tolerance.
        assertFailsWith<IllegalStateException> { GaussJacobi.computeReference(3, -0.5, 0.0, maxNewtonIterations = 1) }
    }

    @Test fun lnGammaMatchesOracle() {
        // Relative error with a unit floor: ln Γ vanishes at x = 1 and x = 2.
        val grid = (1..1000).map { it * 0.05 } + listOf(1e-12, 1e-6, 1e-3, 0.1, 0.25, 0.49, 0.51, 1.4616321449683623)
        for (x in grid) {
            val ref = Gamma.logGamma(x)
            val got = SpecialFunctions.lnGamma(x)
            assertTrue(abs(got - ref) <= 1e-14 * maxOf(1.0, abs(ref)), "x = $x: $got vs $ref")
        }
    }
}
