package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Тесты порядков сходимости по таблице погрешностей: orders (порядок log2),
 * constCh (константа E_h/h^p).
 */
@Tag("fast")
class ConvergenceRatesTest {
    /** orders: для геометрически убывающих ошибок (деление на 2) порядок = 1, последний = NaN. */
    @Test fun ordersGeometricHalving() {
        val errs = listOf(1.0, 0.5, 0.25)
        val p = orders(errs)
        assertEquals(3, p.size)
        assertEquals(1.0, p[0], 1e-12)
        assertEquals(1.0, p[1], 1e-12)
        assertTrue(p[2].isNaN()) // последний элемент без следующего -> NaN
    }

    /** orders: убывание в 4 раза даёт порядок 2. */
    @Test fun ordersQuarteringIsOrderTwo() {
        val p = orders(listOf(1.0, 0.25))
        assertEquals(2.0, p[0], 1e-12)
    }

    /**
     * orders: нулевая погрешность (совпадение с точным решением) — порядок не определён,
     * а не «бесконечен»: раньше log2(E/0) давал +Inf, log2(0/E) — -Inf.
     */
    @Test fun ordersUndefinedOnZeroError() {
        assertTrue(orders(listOf(1.0, 0.0))[0].isNaN(), "E_{h/2}=0 -> порядок не определён")
        assertTrue(orders(listOf(0.0, 1.0))[0].isNaN(), "E_h=0 -> порядок не определён")
        assertTrue(orders(listOf(0.0, 0.0))[0].isNaN(), "обе нулевые -> порядок не определён")
        // отрицательная «погрешность» бессмысленна и тоже даёт NaN, а не log от отрицательного
        assertTrue(orders(listOf(-1.0, 1.0))[0].isNaN(), "отрицательная погрешность -> NaN")
        // положительные значения рядом с нулём считаются по-прежнему
        assertEquals(1.0, orders(listOf(1.0, 0.5, 0.0))[0], 1e-12)
        assertTrue(orders(listOf(1.0, 0.5, 0.0))[1].isNaN())
    }

    /** constCh: C = E_h / h^p — обратный к определению. */
    @Test fun constChDefinition() {
        val eh = 0.08; val h = 0.25; val pp = 2.0
        val c = constCh(eh, h, pp)
        assertEquals(eh / (h * h), c, 1e-12)
        // обратная сверка: c*h^p = eh
        assertTrue(abs(c * h * h - eh) < 1e-12)
    }
}
