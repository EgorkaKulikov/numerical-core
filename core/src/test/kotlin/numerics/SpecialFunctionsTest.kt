package numerics

import org.hipparchus.special.Beta
import org.hipparchus.special.Erf
import org.hipparchus.special.Gamma
import org.junit.jupiter.api.Tag
import java.math.BigDecimal
import java.math.MathContext
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cosh
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

@Tag("fast")
class SpecialFunctionsTest {
    private fun relativeError(got: Double, ref: Double): Double = abs(got - ref) / abs(ref)

    /** Asserts the largest error of a group and prints it (the value lands in the JUnit XML report). */
    private fun assertMaxError(name: String, errors: List<Double>, tolerance: Double) {
        val max = errors.max()
        println("$name: max error $max over ${errors.size} points")
        assertTrue(max <= tolerance, "$name: max error $max exceeds $tolerance")
    }

    /** Same as [assertMaxError] over [points], skipping points where [error] is null; prints the worst point. */
    private fun <T> assertMaxErrorAt(name: String, points: List<T>, tolerance: Double, error: (T) -> Double?) {
        val measured = points.mapNotNull { point -> error(point)?.let { point to it } }
        val (worst, max) = measured.maxBy { it.second }
        println("$name: max error $max at $worst over ${measured.size} points")
        assertTrue(max <= tolerance, "$name: max error $max at $worst exceeds $tolerance")
    }

    // ---------- gamma and lnGamma ----------

    private val mc34 = MathContext.DECIMAL128

    /**
     * Γ at an exact decimal argument `x > 0`, 34 digits: the recurrence `Γ(x) = Γ(r)·(x − 1)…(x − k)` down to
     * `r = x − k ∈ (1.5, 2.5]` (exact in decimal), where Hipparchus `Gamma.gamma` uses its `invGamma1pm1` branch
     * (a few ulps), with the first-order correction `ψ(r)·(r − fl(r))` for the rounding of `r` to double.
     * For `x < 1.5` the recurrence runs upward: `Γ(x) = Γ(x + k)/(x(x + 1)…(x + k − 1))`.
     */
    private fun gammaReferenceBig(x: BigDecimal): BigDecimal {
        val k = BigDecimal(x.toDouble() - 2.5).setScale(0, java.math.RoundingMode.CEILING).toInt()
        val r = x.subtract(BigDecimal(k))
        val rd = r.toDouble()
        val base = BigDecimal(Gamma.gamma(rd)).multiply(BigDecimal.ONE.add(BigDecimal(Gamma.digamma(rd)).multiply(r.subtract(BigDecimal(rd)))), mc34)
        var product = BigDecimal.ONE
        if (k >= 0) {
            for (j in 1..k) product = product.multiply(x.subtract(BigDecimal(j)), mc34)
            return base.multiply(product, mc34)
        }
        for (j in 0 until -k) product = product.multiply(x.add(BigDecimal(j)), mc34)
        return base.divide(product, mc34)
    }

    private fun gammaReference(x: Double): Double = gammaReferenceBig(BigDecimal(x)).toDouble()

    private val gammaGrid = listOf(1e-300, 1e-100, 1e-12, 1e-6, 1e-3, 0.1, 0.25, 0.49, 0.5, 0.51, 1.4616321449683623, 170.5, 171.0) +
        (1..3420).map { it * 0.05 } + (0..1709).map { it * 0.1 + 0.0371 }

    @Test fun gammaMatchesReference() {
        assertMaxErrorAt("gamma vs 34-digit reference", gammaGrid, 1e-14) { relativeError(SpecialFunctions.gamma(it), gammaReference(it)) }
    }

    @Test fun gammaMatchesHipparchus() {
        // Hipparchus overflows above x ≈ 141 (pow(x + 5.24, x + 0.5)); the comparison stops at 140.
        val grid = gammaGrid.filter { it <= 140.0 }
        // Hipparchus switches to a Lanczos formula above 20 whose own error reaches 7e-14: the 1e-14 bound applies to
        // the disagreement beyond the oracle's own error against the 34-digit reference.
        assertMaxErrorAt("Hipparchus gamma vs 34-digit reference", grid, 1.0) { relativeError(Gamma.gamma(it), gammaReference(it)) }
        assertMaxErrorAt("gamma vs Hipparchus (raw)", grid, 1.0) { relativeError(SpecialFunctions.gamma(it), Gamma.gamma(it)) }
        assertMaxErrorAt("gamma vs Hipparchus beyond its own error", grid, 1e-14) {
            relativeError(SpecialFunctions.gamma(it), Gamma.gamma(it)) - relativeError(Gamma.gamma(it), gammaReference(it))
        }
    }

    @Test fun gammaIsExactFactorialAndHalf() {
        var factorial = 1.0
        for (n in 1..20) {
            if (n > 1) factorial *= n - 1
            assertTrue(relativeError(SpecialFunctions.gamma(n.toDouble()), factorial) <= 1e-15, "n = $n")
        }
        assertEquals(sqrt(PI), SpecialFunctions.gamma(0.5), 1e-15 * sqrt(PI))
    }

    @Test fun gammaOverflowAndValidation() {
        assertTrue(SpecialFunctions.gamma(171.6).isFinite())
        assertEquals(Double.POSITIVE_INFINITY, SpecialFunctions.gamma(171.7))
        for (x in listOf(0.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY)) {
            assertFailsWith<IllegalArgumentException> { SpecialFunctions.gamma(x) }
            assertFailsWith<IllegalArgumentException> { SpecialFunctions.lnGamma(x) }
        }
    }

    @Test fun lnGammaMatchesOracle() {
        // Mixed measure: ln Γ vanishes at x = 1 and x = 2.
        val grid = (0..400).map { 10.0.pow(-300.0 + it * 305.0 / 400).coerceAtMost(1e5) } + (1..2000).map { it * 0.025 }
        val errors = grid.map {
            val ref = Gamma.logGamma(it)
            abs(SpecialFunctions.lnGamma(it) - ref) / maxOf(1.0, abs(ref))
        }
        assertMaxError("lnGamma", errors, 1e-14)
    }

    // ---------- beta ----------

    private val betaValues = listOf(1e-3, 0.01, 0.1, 0.25, 1.0 / 3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, PI, 5.0, 7.3, 10.0, 12.5, 15.0, 19.9, 20.0) +
        (1..39).map { it * 0.51 }
    private val betaGrid = betaValues.flatMap { p -> betaValues.map { q -> p to q } }

    /** `B(p, q)` from [gammaReferenceBig] at the exact decimal sum `p + q`. */
    private fun betaReference(p: Double, q: Double): Double {
        val bp = BigDecimal(p)
        val bq = BigDecimal(q)
        return gammaReferenceBig(bp).multiply(gammaReferenceBig(bq), mc34).divide(gammaReferenceBig(bp.add(bq)), mc34).toDouble()
    }

    @Test fun betaMatchesReference() {
        assertMaxErrorAt("beta vs 34-digit reference", betaGrid, 1e-14) { (p, q) -> relativeError(SpecialFunctions.beta(p, q), betaReference(p, q)) }
    }

    @Test fun betaMatchesHipparchus() {
        assertMaxErrorAt("exp(Hipparchus logBeta) vs 34-digit reference", betaGrid, 1.0) { (p, q) ->
            relativeError(exp(Beta.logBeta(p, q)), betaReference(p, q))
        }
        assertMaxErrorAt("beta vs exp(Hipparchus logBeta)", betaGrid, 1e-14) { (p, q) -> relativeError(SpecialFunctions.beta(p, q), exp(Beta.logBeta(p, q))) }
    }

    @Test fun betaLargeArgumentsSymmetryAndValidation() {
        // Above p + q = 170 the error is set by the rounding unit of lnGamma(p + q) ≈ 750–860.
        for ((p, q) in listOf(100.0 to 80.0, 200.0 to 0.5, 0.5 to 200.0)) {
            assertTrue(relativeError(SpecialFunctions.beta(p, q), exp(Beta.logBeta(p, q))) <= 1e-12, "p = $p, q = $q")
        }
        assertEquals(SpecialFunctions.beta(0.3, 7.5), SpecialFunctions.beta(7.5, 0.3))
        val invalid = listOf(0.0 to 1.0, Double.POSITIVE_INFINITY to 1.0, 1.0 to -1.0, 1.0 to Double.POSITIVE_INFINITY, Double.NaN to 1.0)
        for ((p, q) in invalid) assertFailsWith<IllegalArgumentException> { SpecialFunctions.beta(p, q) }
    }

    // ---------- erfc ----------

    private val mc = MathContext(80)
    private val piBig = BigDecimal("3.1415926535897932384626433832795028841971693993751058209749445923078164062862")
    private val sqrtPiBig = piBig.sqrt(mc)
    private val negligible = BigDecimal("1e-90")

    /** `e^y` in 80-digit arithmetic: halving until `|y| ≤ 1e-3`, Taylor series, repeated squaring. */
    private fun expBig(y: BigDecimal): BigDecimal {
        var reduced = y
        var halvings = 0
        while (reduced.abs() > BigDecimal("0.001")) {
            reduced = reduced.divide(BigDecimal(2), mc)
            halvings++
        }
        var term = BigDecimal.ONE
        var sum = BigDecimal.ONE
        var n = 1
        while (term.abs() > negligible) {
            term = term.multiply(reduced, mc).divide(BigDecimal(n), mc)
            sum = sum.add(term, mc)
            n++
        }
        repeat(halvings) { sum = sum.multiply(sum, mc) }
        return sum
    }

    /**
     * Independent `erfc` reference in 80-digit arithmetic: Maclaurin series of `erf` for `|x| < 2`, the Legendre
     * continued fraction of `Γ(1/2, x²)` with 300 levels for `|x| ≥ 2`, `erfc(x) = 2 − erfc(−x)` for `x < 0`.
     */
    private fun erfcReference(x: Double): Double {
        val xb = BigDecimal(abs(x))
        val xx = xb.multiply(xb)
        val value = if (abs(x) < 2.0) {
            var power = xb
            var sum = xb
            var n = 0
            while (true) {
                n++
                power = power.multiply(xx, mc).negate().divide(BigDecimal(n), mc)
                val term = power.divide(BigDecimal(2 * n + 1), mc)
                sum = sum.add(term, mc)
                if (term.abs() < negligible) break
            }
            BigDecimal.ONE.subtract(sum.multiply(BigDecimal(2)).divide(sqrtPiBig, mc), mc)
        } else {
            val levels = 300
            var tail = xx.add(BigDecimal(0.5 + 2.0 * levels))
            for (k in levels downTo 1) {
                tail = xx.add(BigDecimal(0.5 + 2.0 * (k - 1))).subtract(BigDecimal(k * (k - 0.5)).divide(tail, mc), mc)
            }
            xb.multiply(expBig(xx.negate()), mc).divide(sqrtPiBig, mc).divide(tail, mc)
        }
        return (if (x < 0.0) BigDecimal(2).subtract(value, mc) else value).toDouble()
    }

    private val erfcGrid = (0..2112).map { -6.0 + it / 64.0 } + (0..890).map { -6.0 + it * 0.0371 } + listOf(26.5, 26.55)

    @Test fun erfcMatchesReference() {
        assertMaxErrorAt("erfc vs 80-digit reference", erfcGrid, 1e-14) { x ->
            val ref = erfcReference(x)
            if (ref >= 1e-300) relativeError(SpecialFunctions.erfc(x), ref) else null
        }
    }

    @Test fun erfcMatchesHipparchus() {
        assertMaxErrorAt("Hipparchus erfc vs 80-digit reference", erfcGrid, 1.0) { x ->
            val ref = erfcReference(x)
            if (ref >= 1e-300) relativeError(Erf.erfc(x), ref) else null
        }
        assertMaxErrorAt("erfc vs Hipparchus (raw)", erfcGrid, 1.0) { x ->
            val ref = Erf.erfc(x)
            if (ref >= 1e-300) relativeError(SpecialFunctions.erfc(x), ref) else null
        }
        // Hipparchus erfc (regularized Γ(1/2, x²) to 1e-15) loses accuracy for large x; as for gamma, the 1e-14 bound
        // applies to the disagreement beyond the oracle's own error against the 80-digit reference.
        assertMaxErrorAt("erfc vs Hipparchus beyond its own error", erfcGrid, 1e-14) { x ->
            val ref = erfcReference(x)
            if (ref >= 1e-300) relativeError(SpecialFunctions.erfc(x), Erf.erfc(x)) - relativeError(Erf.erfc(x), ref) else null
        }
    }

    @Test fun erfcSpecialValuesAndValidation() {
        assertEquals(1.0, SpecialFunctions.erfc(0.0))
        assertEquals(0.0, SpecialFunctions.erfc(27.5))
        assertEquals(0.0, SpecialFunctions.erfc(Double.POSITIVE_INFINITY))
        assertEquals(2.0, SpecialFunctions.erfc(Double.NEGATIVE_INFINITY))
        assertEquals(2.0, SpecialFunctions.erfc(-30.0))
        assertFailsWith<IllegalArgumentException> { SpecialFunctions.erfc(Double.NaN) }
    }

    @Test fun erfcContinuedFractionIsConverged() {
        val errors = listOf(1.0, 1.3, 2.0, 4.0, 10.0, 26.0).map { x ->
            val terms = SpecialFunctions.continuedFractionTerms(x)
            relativeError(SpecialFunctions.erfcContinuedFraction(x, terms), SpecialFunctions.erfcContinuedFraction(x, 4 * terms))
        }
        assertMaxError("erfc continued fraction truncation", errors, 1e-15)
    }

    // ---------- Mittag-Leffler ----------

    private val mittagLefflerGrid = (0..300).map { it * 0.01 }

    @Test fun mittagLefflerMatchesClosedForms() {
        val ml = SpecialFunctions::mittagLeffler
        assertMaxError("E_1(z) vs exp(z)", mittagLefflerGrid.map { relativeError(ml(1.0, it), exp(it)) }, 1e-13)
        assertMaxError("E_2(z) vs cosh(sqrt z)", mittagLefflerGrid.map { relativeError(ml(2.0, it), cosh(sqrt(it))) }, 1e-13)
        assertMaxError(
            "E_1/2(z) vs exp(z^2) erfc(-z), own erfc",
            mittagLefflerGrid.map { relativeError(ml(0.5, it), exp(it * it) * SpecialFunctions.erfc(-it)) },
            1e-13,
        )
        assertMaxError(
            "E_1/2(z) vs exp(z^2) erfc(-z), Hipparchus erfc",
            mittagLefflerGrid.map { relativeError(ml(0.5, it), exp(it * it) * Erf.erfc(-it)) },
            1e-13,
        )
    }

    @Test fun mittagLefflerEdgesAndValidation() {
        assertEquals(1.0, SpecialFunctions.mittagLeffler(0.7, 0.0))
        assertTrue(relativeError(SpecialFunctions.mittagLeffler(1.0, 10.0), exp(10.0)) <= 1e-13)
        assertTrue(relativeError(SpecialFunctions.mittagLeffler(2.0, 10.0), cosh(sqrt(10.0))) <= 1e-13)
        assertEquals(Double.POSITIVE_INFINITY, SpecialFunctions.mittagLeffler(1.0 / 3, 10.0))
        val invalid = listOf(0.33 to 1.0, 2.01 to 1.0, Double.NaN to 1.0, 1.0 to -0.1, 1.0 to 10.01, 1.0 to Double.NaN)
        for ((beta, z) in invalid) assertFailsWith<IllegalArgumentException> { SpecialFunctions.mittagLeffler(beta, z) }
    }

    /**
     * Residual of `u(t) − ∫_0^t (t − s)^(−alpha) u(s) ds = 1`; the partition is refined geometrically toward 0,
     * breakpoints `t·2^(−k)`, `k = 40..0`, because `u − 1 ~ t^(1 − alpha)` is not smooth at 0.
     */
    private fun abelResidual(alpha: Double, t: Double, u: (Double) -> Double): Double {
        val breakpoints = DoubleArray(42) { i -> if (i == 0) 0.0 else t * 2.0.pow(i - 41) }
        return u(t) - AlgebraicSingularQuadrature(alpha).integrate(breakpoints, t, u) - 1.0
    }

    @Test fun mittagLefflerSolvesAbelEquations() {
        val times = listOf(0.1, 0.5, 1.0)
        // V-c: u(t) = E_{2/3}(Γ(2/3)·t^{2/3}), kernel (t − s)^(−1/3).
        val gammaTwoThirds = SpecialFunctions.gamma(2.0 / 3)
        val vc = { s: Double -> SpecialFunctions.mittagLeffler(2.0 / 3, gammaTwoThirds * s.pow(2.0 / 3)) }
        assertMaxError("V-c residual", times.map { abs(abelResidual(1.0 / 3, it, vc)) }, 1e-12)
        // V-a: u(t) = E_{1/2}(√(πt)) = e^{πt}·erfc(−√(πt)), kernel (t − s)^(−1/2).
        val va = { s: Double -> SpecialFunctions.mittagLeffler(0.5, sqrt(PI * s)) }
        assertMaxError("V-a residual", times.map { abs(abelResidual(0.5, it, va)) }, 1e-12)
        assertMaxError("V-a closed form", times.map { relativeError(va(it), exp(PI * it) * SpecialFunctions.erfc(-sqrt(PI * it))) }, 1e-13)
    }
}
