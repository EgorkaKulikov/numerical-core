package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests of the reliability threshold for measured quantities: classification by [measured] and
 * refusal to compute a ratio/convergence order from values at machine-noise level.
 *
 * The reference numbers come from the real case that motivated the mechanism: the values
 * 7.028e-14, 3.542e-14, 6.539e-14, 3.275e-14 ended up in a table and an error ratio was
 * computed from them, although all of them are below the 1e-13 threshold.
 */
@Tag("fast")
class ReliabilityTest {

    /** A value above the threshold is reliable, below it is flagged as noise. */
    @Test fun measuredClassifiesByThreshold() {
        val ok = measured(1.5e-9)
        assertTrue(ok is Measured.Reliable)
        assertEquals(1.5e-9, ok.value, 0.0)

        val noise = measured(7.028e-14)
        assertTrue(noise is Measured.AtNoiseLevel)
        assertEquals(MACHINE_NOISE_THRESHOLD, (noise as Measured.AtNoiseLevel).threshold, 0.0)
        assertEquals(7.028e-14, noise.value, 0.0)
    }

    /** The boundary is inclusive: exactly the threshold is a reliable value. */
    @Test fun measuredIncludesThresholdItself() {
        assertTrue(measured(MACHINE_NOISE_THRESHOLD) is Measured.Reliable)
        assertTrue(measured(-MACHINE_NOISE_THRESHOLD) is Measured.Reliable)
        assertTrue(measured(MACHINE_NOISE_THRESHOLD / 2) is Measured.AtNoiseLevel)
    }

    /** Non-finite values are not measurements: NaN and infinities are unreliable. */
    @Test fun measuredRejectsNonFinite() {
        assertTrue(measured(Double.NaN) is Measured.AtNoiseLevel)
        assertTrue(measured(Double.POSITIVE_INFINITY) is Measured.AtNoiseLevel)
        assertTrue(measured(0.0) is Measured.AtNoiseLevel)
    }

    /** A non-positive threshold is a contract violation: it would mean no check at all. */
    @Test fun measuredRejectsNonPositiveThreshold() {
        assertFailsWith<IllegalArgumentException> { measured(1.0, threshold = 0.0) }
        assertFailsWith<IllegalArgumentException> { measured(1.0, threshold = -1e-13) }
    }

    /** The ratio of reliable values is computed; a ratio involving noise is not. */
    @Test fun ratioRefusesNoise() {
        assertEquals(2.0, ratio(measured(4e-6), measured(2e-6))!!, 1e-12)
        assertNull(ratio(measured(7.028e-14), measured(3.542e-14)))
        assertNull(ratio(measured(4e-6), measured(3.275e-14)))
        assertNull(ratio(measured(6.539e-14), measured(4e-6)))
    }

    /** Exactly the case that motivated the mechanism: 7.028e-14 / 3.542e-14 is not a number. */
    @Test fun realCaseFromTableIsRejected() {
        val printed = listOf(7.028e-14, 3.542e-14, 6.539e-14, 3.275e-14)
        for (v in printed) assertTrue(measured(v) is Measured.AtNoiseLevel, "$v is below the threshold")
        assertNull(ratio(measured(printed[0]), measured(printed[1])))
        assertTrue(reliableOrders(printed).all { it == null })
    }

    /** The convergence order from reliable values equals log2 of the ratio. */
    @Test fun orderOrNullMatchesLog2() {
        assertEquals(2.0, orderOrNull(measured(4e-6), measured(1e-6))!!, 1e-12)
        assertEquals(1.0, orderOrNull(measured(2e-6), measured(1e-6))!!, 1e-12)
    }

    /** A negative ratio has no logarithm: the order is undefined. */
    @Test fun orderOrNullRejectsNonPositiveRatio() {
        assertNull(orderOrNull(measured(-4e-6), measured(1e-6)))
    }

    /** An order is never computed from noise, whichever side it is on. */
    @Test fun orderOrNullRejectsNoise() {
        assertNull(orderOrNull(measured(1e-6), measured(3.275e-14)))
        assertNull(orderOrNull(measured(3.275e-14), measured(1e-6)))
    }

    /** On reliable data reliableOrders agrees with the existing orders. */
    @Test fun reliableOrdersAgreeWithOrdersOnReliableData() {
        val errs = listOf(1e-3, 2.5e-4, 6.25e-5)
        val ref = orders(errs)
        val rel = reliableOrders(errs)
        assertEquals(errs.size, rel.size)
        assertEquals(ref[0], rel[0]!!, 1e-12)
        assertEquals(ref[1], rel[1]!!, 1e-12)
        assertNull(rel[2]) // the last row has nothing to compare against
        assertTrue(ref[2].isNaN())
    }

    /** Mixed table: the order is computed only where both adjacent values are reliable. */
    @Test fun reliableOrdersStopAtNoiseBoundary() {
        val errs = listOf(1e-6, 2.5e-7, 5e-14)
        val rel = reliableOrders(errs)
        assertEquals(2.0, rel[0]!!, 1e-12)
        assertNull(rel[1]) // 5e-14 is below the threshold
        assertNull(rel[2])
    }

    /** The threshold is configurable: with a looser threshold the same data becomes reliable. */
    @Test fun reliableOrdersRespectCustomThreshold() {
        val errs = listOf(4e-14, 1e-14)
        assertNull(reliableOrders(errs)[0])
        assertEquals(2.0, reliableOrders(errs, threshold = 1e-15)[0]!!, 1e-12)
    }

    /** reliableConstCh: equals constCh on a reliable value, null on noise. */
    @Test fun reliableConstChAgreesWithConstCh() {
        val eh = 0.08
        val h = 0.25
        assertEquals(constCh(eh, h, 2.0), reliableConstCh(measured(eh), h, 2.0)!!, 1e-12)
        assertNull(reliableConstCh(measured(3.542e-14), h, 2.0))
    }
}
