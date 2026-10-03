package numerics

import org.hipparchus.special.Beta
import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@Tag("fast")
class AlgebraicSingularQuadratureTest {
    private val alphas = listOf(1.0 / 3, 0.5, 2.0 / 3, 0.9)
    private val unitPartition = doubleArrayOf(0.0, 0.25, 0.5, 0.75, 1.0)
    private val smooth = { s: Double -> exp(s) + cos(3 * s) }

    private fun relErr(actual: Double, expected: Double): Double = abs(actual - expected) / abs(expected)

    private fun checkRel(expected: Double, actual: Double, tol: Double, label: String): Double {
        val err = relErr(actual, expected)
        assertTrue(err <= tol, "$label: expected $expected, got $actual, relative error $err > $tol")
        return err
    }

    /** `∫_a^b |t − s|^(−alpha) f(s) ds` for `a ≤ t ≤ b`: split at `t`, Gauss–Jacobi with 40 nodes on each side. */
    private fun splitReference(alpha: Double, t: Double, a: Double, b: Double, f: (Double) -> Double): Double {
        var sum = 0.0
        if (t > a) sum += GaussJacobi(40, -alpha, 0.0).integrate(a, t, f)
        if (t < b) sum += GaussJacobi(40, 0.0, -alpha).integrate(t, b, f)
        return sum
    }

    /**
     * `∫_a^b |t − s|^(−alpha) f(s) ds` for `t` outside `[a, b]`: Gauss–Legendre with 30 nodes in the distance
     * `u = |s − t|` on a partition graded geometrically (ratio 1.5) from the nearest end.
     */
    private fun gradedReference(alpha: Double, t: Double, a: Double, b: Double, f: (Double) -> Double): Double {
        val left = t < a
        val near = if (left) a - t else t - b
        val far = if (left) b - t else t - a
        val points = mutableListOf(near)
        while (points.last() * 1.5 < far) points += points.last() * 1.5
        points += far
        return GaussLegendre(30).integrate(points.toDoubleArray()) { u ->
            u.pow(-alpha) * f(if (left) t + u else t - u)
        }
    }

    @Test
    fun `alpha zero reduces to composite Gauss-Legendre`() {
        val partition = doubleArrayOf(-1.0, -0.2, 0.3, 1.5, 2.0)
        var maxErr = 0.0
        for (m in listOf(1, 5, 12)) {
            val expected = GaussLegendre(m).integrate(partition, smooth)
            for (t in listOf(-3.0, -1.0, 0.3, 0.7, 2.0, 10.0)) {
                val actual = AlgebraicSingularQuadrature(0.0, m).integrate(partition, t, smooth)
                maxErr = max(maxErr, checkRel(expected, actual, 1e-15, "m=$m, t=$t"))
            }
        }
        println("A2 alpha=0 vs GaussLegendre: max rel err = $maxErr")
    }

    @Test
    fun `constructor parameters are exposed with default node count`() {
        val q = AlgebraicSingularQuadrature(0.5)
        assertEquals(0.5, q.alpha)
        assertEquals(12, q.nodesPerSub)
    }

    @Test
    fun `Volterra moments on partitions ending at t match the Beta function`() {
        val partitions = listOf(
            DoubleArray(6) { 0.7 * it / 5 }.also { it[5] = 0.7 },
            doubleArrayOf(0.0, 0.2, 0.4, 0.6, 0.73),
            doubleArrayOf(0.0, 0.5, 0.65, 0.69, 0.7),
            doubleArrayOf(0.0, 1.0),
            doubleArrayOf(0.0, 1e-3, 2.5),
        )
        var maxErr = 0.0
        for (partition in partitions) {
            val t = partition.last()
            for (alpha in alphas) {
                val q = AlgebraicSingularQuadrature(alpha)
                for (j in 0..4) {
                    val exact = t.pow(j + 1 - alpha) * exp(Beta.logBeta(1 - alpha, j + 1.0))
                    val actual = q.integrate(partition, t) { s -> s.pow(j) }
                    maxErr = max(maxErr, checkRel(exact, actual, 1e-13, "t=$t, alpha=$alpha, j=$j"))
                }
            }
        }
        println("A2 Volterra moments: max rel err = $maxErr")
    }

    @Test
    fun `two-sided kernel with t inside the partition matches closed forms`() {
        val q = AlgebraicSingularQuadrature(0.5)
        var maxErr = 0.0
        var maxRefErr = 0.0
        for (t in listOf(0.0, 1e-6, 0.3, 0.5, 1.0)) {
            val exactConst = 2 * (sqrt(t) + sqrt(1 - t))
            // ∫_0^t (t − s)^(−1/2) s ds + ∫_t^1 (s − t)^(−1/2) s ds
            val exactLinear = 4.0 / 3 * t.pow(1.5) + 2.0 / 3 * (1 - t).pow(1.5) + 2 * t * sqrt(1 - t)
            maxRefErr = max(maxRefErr, checkRel(exactLinear, splitReference(0.5, t, 0.0, 1.0) { it }, 1e-14, "ref t=$t"))
            maxErr = max(maxErr, checkRel(exactConst, q.integrate(unitPartition, t) { 1.0 }, 1e-13, "1, t=$t"))
            maxErr = max(maxErr, checkRel(exactLinear, q.integrate(unitPartition, t) { it }, 1e-13, "s, t=$t"))
        }
        println("A2 two-sided closed forms: max rel err = $maxErr (reference vs closed form $maxRefErr)")
    }

    @Test
    fun `two-sided kernel with smooth f matches split Gauss-Jacobi reference`() {
        var maxErr = 0.0
        for (alpha in alphas) {
            val q = AlgebraicSingularQuadrature(alpha)
            for (t in listOf(0.0, 1e-6, 0.3, 0.5, 0.999, 1.0)) {
                val expected = splitReference(alpha, t, 0.0, 1.0, smooth)
                val actual = q.integrate(unitPartition, t, smooth)
                maxErr = max(maxErr, checkRel(expected, actual, 1e-13, "alpha=$alpha, t=$t"))
            }
        }
        println("A2 two-sided smooth f: max rel err = $maxErr")
    }

    @Test
    fun `t outside the partition matches graded Gauss-Legendre reference`() {
        var maxErr = 0.0
        for (alpha in alphas) {
            val q = AlgebraicSingularQuadrature(alpha)
            for (t in listOf(1 + 1e-6, -1e-4, 5.0, -5.0, 1.1)) {
                val expected = gradedReference(alpha, t, 0.0, 1.0, smooth)
                val actual = q.integrate(unitPartition, t, smooth)
                maxErr = max(maxErr, checkRel(expected, actual, 1e-12, "alpha=$alpha, t=$t"))
            }
        }
        println("A2 t outside the partition: max rel err = $maxErr")
    }

    @Test
    fun `piece cap bounds the work when t almost touches a cell`() {
        val q = AlgebraicSingularQuadrature(0.5)
        val cell = doubleArrayOf(0.0, 1.0)

        // Just above the cap threshold: the refinement finishes on its own and stays accurate.
        val dFine = Math.scalb(1.0, -58)
        var calls = 0
        val fine = q.integrate(cell, -dFine) { calls++; 1.0 }
        assertTrue(calls < AlgebraicSingularQuadrature.MAX_PIECES * q.nodesPerSub)
        val fineErr = checkRel(2 * (sqrt(1 + dFine) - sqrt(dFine)), fine, 1e-13, "d=2^-58")

        // Below it: exactly MAX_PIECES pieces, the last one too long, accuracy degrades gracefully.
        val dCapped = Math.scalb(1.0, -62)
        calls = 0
        val capped = q.integrate(cell, -dCapped) { calls++; 1.0 }
        assertEquals(AlgebraicSingularQuadrature.MAX_PIECES * q.nodesPerSub, calls)
        val cappedErr = checkRel(2 * (sqrt(1 + dCapped) - sqrt(dCapped)), capped, 1e-6, "d=2^-62")
        println("A2 piece cap: rel err below cap (d=2^-58) = $fineErr, capped (d=2^-62) = $cappedErr")
    }

    @Test
    fun `invalid constructor arguments are rejected`() {
        assertFailsWith<IllegalArgumentException> { AlgebraicSingularQuadrature(-0.1) }
        assertFailsWith<IllegalArgumentException> { AlgebraicSingularQuadrature(1.0) }
        assertFailsWith<IllegalArgumentException> { AlgebraicSingularQuadrature(Double.NaN) }
        val e = assertFailsWith<IllegalArgumentException> { AlgebraicSingularQuadrature(0.5, 0) }
        assertTrue(e.message!!.contains("at least 1"))
    }

    @Test
    fun `invalid integrate arguments are rejected`() {
        val q = AlgebraicSingularQuadrature(0.5)
        val f = { _: Double -> 1.0 }
        assertFailsWith<IllegalArgumentException> { q.integrate(doubleArrayOf(), 0.0, f) }
        assertFailsWith<IllegalArgumentException> { q.integrate(doubleArrayOf(0.0), 0.0, f) }
        assertFailsWith<IllegalArgumentException> { q.integrate(doubleArrayOf(0.0, 0.0), 0.0, f) }
        assertFailsWith<IllegalArgumentException> { q.integrate(doubleArrayOf(0.0, 1.0, 0.5), 0.0, f) }
        assertFailsWith<IllegalArgumentException> { q.integrate(doubleArrayOf(Double.NaN, 1.0), 0.0, f) }
        assertFailsWith<IllegalArgumentException> { q.integrate(doubleArrayOf(0.0, Double.POSITIVE_INFINITY), 0.0, f) }
        assertFailsWith<IllegalArgumentException> { q.integrate(doubleArrayOf(Double.NEGATIVE_INFINITY, 0.0), 0.0, f) }
        assertFailsWith<IllegalArgumentException> { q.integrate(unitPartition, Double.NaN, f) }
        assertFailsWith<IllegalArgumentException> { q.integrate(unitPartition, Double.POSITIVE_INFINITY, f) }
        assertFailsWith<IllegalArgumentException> {
            AlgebraicSingularQuadrature(0.0).integrate(doubleArrayOf(1.0, 0.0), 0.5, f)
        }
    }
}
