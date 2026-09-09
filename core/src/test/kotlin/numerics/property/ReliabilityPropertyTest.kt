package numerics.property

import net.jqwik.api.Arbitraries
import net.jqwik.api.Arbitrary
import net.jqwik.api.ForAll
import net.jqwik.api.Property
import net.jqwik.api.Provide
import net.jqwik.api.Tag
import net.jqwik.api.constraints.DoubleRange
import net.jqwik.api.constraints.IntRange
import net.jqwik.api.constraints.Scale
import numerics.MACHINE_NOISE_THRESHOLD
import numerics.Measured
import numerics.measured
import numerics.orders
import numerics.ratio
import numerics.reliableOrders
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import kotlin.math.abs
import kotlin.math.pow

@Tag("fast")
class ReliabilityPropertyTest {

    /** Любые double, включая NaN, ±∞, ±0 и значения вблизи порога шума. */
    @Provide
    fun anyDouble(): Arbitrary<Double> = Arbitraries.oneOf(
        Arbitraries.doubles(),
        Arbitraries.doubles().between(-1e-12, 1e-12).ofScale(16),
        Arbitraries.of(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, 0.0, -0.0, MACHINE_NOISE_THRESHOLD, -MACHINE_NOISE_THRESHOLD),
    )

    @Provide fun thresholds(): Arbitrary<Double> = Arbitraries.of(MACHINE_NOISE_THRESHOLD, 1e-16, 1e-8, 1.0)

    /** Положительные погрешности: обычные, шумовые и ровно нулевые. */
    @Provide
    fun errorLists(): Arbitrary<List<Double>> = Arbitraries.oneOf(
        Arbitraries.doubles().between(1e-12, 10.0).ofScale(14),
        Arbitraries.doubles().between(0.0, 1e-13).ofScale(16),
        Arbitraries.of(0.0, 1e-13, 5e-14),
    ).list().ofMinSize(1).ofMaxSize(8)

    @Property(tries = 100)
    fun `measured достоверно тогда и только тогда, когда значение конечно и не ниже порога`(@ForAll("anyDouble") v: Double, @ForAll("thresholds") thr: Double) {
        val m = measured(v, thr)
        val expectReliable = v.isFinite() && abs(v) >= thr
        assertEquals(expectReliable, m is Measured.Reliable) { "measured($v, $thr) = $m" }
        when (m) {
            is Measured.Reliable -> assertEquals(v.toRawBits(), m.value.toRawBits())
            is Measured.AtNoiseLevel -> { assertEquals(v.toRawBits(), m.value.toRawBits()); assertEquals(thr, m.threshold) }
        }
    }

    @Property(tries = 100)
    fun `reliableOrders обнуляет порядок на шуме и повторяет orders иначе`(@ForAll("errorLists") errs: List<Double>) {
        val rel = reliableOrders(errs)
        val plain = orders(errs)
        assertEquals(errs.size, rel.size)
        assertNull(rel.last())
        for (i in 0 until errs.size - 1) {
            val noisy = measured(errs[i]) is Measured.AtNoiseLevel || measured(errs[i + 1]) is Measured.AtNoiseLevel
            if (noisy) {
                assertNull(rel[i]) { "позиция $i: ожидался null на шуме, получено ${rel[i]}" }
            } else {
                assertEquals(plain[i].toRawBits(), rel[i]!!.toRawBits()) { "позиция $i: ${rel[i]} vs ${plain[i]}" }
            }
        }
    }

    @Property(tries = 100)
    fun `orders восстанавливает показатель геометрической последовательности`(
        @ForAll @DoubleRange(min = 0.5, max = 4.0) p: Double,
        @ForAll @DoubleRange(min = 1e-3, max = 10.0) @Scale(6) c: Double,
        @ForAll @IntRange(min = 3, max = 8) size: Int,
    ) {
        val errs = List(size) { i -> c * 2.0.pow(-p * i) }
        val got = orders(errs)
        for (i in 0 until size - 1) assertTrue(abs(got[i] - p) <= 1e-10) { "позиция $i: ${got[i]} vs $p" }
        assertTrue(got.last().isNaN())
    }

    @Property(tries = 100)
    fun `ratio определено только для двух достоверных значений`(@ForAll("anyDouble") a: Double, @ForAll("anyDouble") b: Double) {
        val ma = measured(a)
        val mb = measured(b)
        val r = ratio(ma, mb)
        if (ma is Measured.AtNoiseLevel || mb is Measured.AtNoiseLevel) {
            assertNull(r)
        } else {
            assertEquals((a / b).toRawBits(), r!!.toRawBits())
        }
    }
}
