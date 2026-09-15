package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests of convergence orders computed from an error table: orders (log2 order),
 * constCh (the constant E_h/h^p).
 */
@Tag("fast")
class ConvergenceRatesTest {
    /** orders: for geometrically decreasing errors (halving) the order is 1, the last entry is NaN. */
    @Test fun ordersGeometricHalving() {
        val errs = listOf(1.0, 0.5, 0.25)
        val p = orders(errs)
        assertEquals(3, p.size)
        assertEquals(1.0, p[0], 1e-12)
        assertEquals(1.0, p[1], 1e-12)
        assertTrue(p[2].isNaN()) // last element has no successor -> NaN
    }

    /** orders: a fourfold decrease gives order 2. */
    @Test fun ordersQuarteringIsOrderTwo() {
        val p = orders(listOf(1.0, 0.25))
        assertEquals(2.0, p[0], 1e-12)
    }

    /**
     * orders: a zero error (exact agreement with the true solution) leaves the order undefined,
     * not "infinite": previously log2(E/0) gave +Inf and log2(0/E) gave -Inf.
     */
    @Test fun ordersUndefinedOnZeroError() {
        assertTrue(orders(listOf(1.0, 0.0))[0].isNaN(), "E_{h/2}=0 -> order undefined")
        assertTrue(orders(listOf(0.0, 1.0))[0].isNaN(), "E_h=0 -> order undefined")
        assertTrue(orders(listOf(0.0, 0.0))[0].isNaN(), "both zero -> order undefined")
        // a negative "error" is meaningless and also yields NaN rather than a log of a negative number
        assertTrue(orders(listOf(-1.0, 1.0))[0].isNaN(), "negative error -> NaN")
        // positive values next to the zero are still computed
        assertEquals(1.0, orders(listOf(1.0, 0.5, 0.0))[0], 1e-12)
        assertTrue(orders(listOf(1.0, 0.5, 0.0))[1].isNaN())
    }

    /** constCh: C = E_h / h^p — the inverse of the definition. */
    @Test fun constChDefinition() {
        val eh = 0.08; val h = 0.25; val pp = 2.0
        val c = constCh(eh, h, pp)
        assertEquals(eh / (h * h), c, 1e-12)
        // reverse check: c*h^p = eh
        assertTrue(abs(c * h * h - eh) < 1e-12)
    }
}
