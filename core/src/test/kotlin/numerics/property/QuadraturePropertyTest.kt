package numerics.property

import net.jqwik.api.Arbitraries
import net.jqwik.api.Arbitrary
import net.jqwik.api.Assume
import net.jqwik.api.ForAll
import net.jqwik.api.Property
import net.jqwik.api.Provide
import net.jqwik.api.Tag
import net.jqwik.api.constraints.DoubleRange
import net.jqwik.api.constraints.IntRange
import net.jqwik.api.constraints.Size
import numerics.GaussLegendre
import numerics.oracle.HipparchusOracle
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.pow

@Tag("fast")
class QuadraturePropertyTest {

    @Provide fun coefficients(): Arbitrary<DoubleArray> = Arbitraries.doubles().between(-1.0, 1.0).array(DoubleArray::class.java).ofSize(24)

    @Provide
    fun breakpoints(): Arbitrary<DoubleArray> =
        Arbitraries.doubles().between(-3.0, 3.0).array(DoubleArray::class.java).ofMinSize(2).ofMaxSize(6).map { it.sortedArray() }

    @Property(tries = 100)
    fun `узлы и веса совпадают с Hipparchus и симметричны`(@ForAll @IntRange(min = 1, max = 64) m: Int) {
        val (x, w) = GaussLegendre.gaussLegendreReference(m)
        val (hx, hw) = HipparchusOracle.gaussLegendre(m)
        assertEquals(m, x.size)
        for (i in 0 until m) {
            assertTrue(abs(x[i] - hx[i]) <= 1e-14) { "узел $i: ${x[i]} vs ${hx[i]}" }
            assertTrue(abs(w[i] - hw[i]) <= 1e-14) { "вес $i: ${w[i]} vs ${hw[i]}" }
            assertTrue(abs(x[i] + x[m - 1 - i]) <= 1e-15) { "узлы не симметричны: $i" }
            if (i > 0) assertTrue(x[i] > x[i - 1]) { "узлы не возрастают: $i" }
        }
        assertTrue(abs(w.sum() - 2.0) <= 1e-14) { "Σw = ${w.sum()}" }
    }

    @Property(tries = 100)
    fun `многочлен степени 2m−1 интегрируется точно`(
        @ForAll @IntRange(min = 1, max = 12) m: Int,
        @ForAll("coefficients") c: DoubleArray,
        @ForAll @DoubleRange(min = -5.0, max = 5.0) lo: Double,
        @ForAll @DoubleRange(min = -5.0, max = 5.0) hi: Double,
    ) {
        Assume.that(lo < hi)
        val deg = 2 * m - 1
        val p = { t: Double -> var s = 0.0; for (k in deg downTo 0) s = s * t + c[k]; s }
        val got = GaussLegendre(m).integrateInterval(lo, hi, p)
        var exact = 0.0
        var scale = 0.0
        for (k in 0..deg) {
            exact += c[k] * (hi.pow(k + 1) - lo.pow(k + 1)) / (k + 1)
            scale += abs(c[k]) * 5.0.pow(k + 1)
        }
        assertTrue(abs(got - exact) <= 1e-12 * (1.0 + scale)) { "m=$m: $got vs $exact, |Δ|=${abs(got - exact)}, scale=$scale" }
    }

    @Property(tries = 100)
    fun `integrate по разбиению равен сумме по отрезкам`(@ForAll("breakpoints") bp: DoubleArray) {
        for (k in 0 until bp.size - 1) Assume.that(bp[k] < bp[k + 1])
        val q = GaussLegendre(6)
        val whole = q.integrate(bp, ::exp)
        var sum = 0.0
        for (k in 0 until bp.size - 1) sum += q.integrateInterval(bp[k], bp[k + 1], ::exp)
        assertTrue(abs(whole - sum) <= 1e-14 * abs(sum)) { "$whole vs $sum" }
    }

    @Property(tries = 100)
    fun `интеграл экспоненты по разбиению совпадает с аналитическим`(@ForAll("breakpoints") bp: DoubleArray) {
        for (k in 0 until bp.size - 1) Assume.that(bp[k] < bp[k + 1])
        val got = GaussLegendre(8).integrate(bp, ::exp)
        val exact = exp(bp.last()) - exp(bp.first())
        // Оценка погрешности 8-точечной формулы Гаусса на одном отрезке длины L для exp:
        // L^17·(8!)^4/(17·(16!)^3)·e^b ≈ 6e-9 при L = 6, b = 3 — то есть 1e-12 относительно
        // достижимо только на коротких отрезках; порог 1e-9 покрывает весь диапазон генератора.
        assertTrue(abs(got - exact) <= 1e-9 * abs(exact)) { "$got vs $exact" }
    }

    @Property(tries = 100)
    fun `перестановка пределов меняет знак побитово`(
        @ForAll @DoubleRange(min = -5.0, max = 5.0) lo: Double,
        @ForAll @DoubleRange(min = -5.0, max = 5.0) hi: Double,
        @ForAll @IntRange(min = 1, max = 12) m: Int,
    ) {
        Assume.that(lo != hi) // при lo == hi обе стороны дают 0.0 и −0.0: равны, но не побитово
        val q = GaussLegendre(m)
        val f = { t: Double -> exp(t) * t }
        assertEquals(q.integrateInterval(lo, hi, f).toRawBits(), (-q.integrateInterval(hi, lo, f)).toRawBits())
    }
}
