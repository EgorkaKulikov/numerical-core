package numerics.functionals

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Тесты порога достоверности величин: классификация [measured] и отказ считать
 * отношение/порядок сходимости по величинам на уровне машинного шума.
 *
 * Эталонные числа взяты из реального случая, ради которого механизм и введён:
 * значения 7.028e-14, 3.542e-14, 6.539e-14, 3.275e-14 попали в таблицу и по ним
 * было посчитано отношение погрешностей, хотя все они ниже порога 1e-13.
 */
@Tag("fast")
class ReliabilityTest {

    /** Величина выше порога достоверна, ниже — помечена как шум. */
    @Test fun measuredClassifiesByThreshold() {
        val ok = measured(1.5e-9)
        assertTrue(ok is Measured.Reliable)
        assertEquals(1.5e-9, ok.value, 0.0)

        val noise = measured(7.028e-14)
        assertTrue(noise is Measured.AtNoiseLevel)
        assertEquals(MACHINE_NOISE_THRESHOLD, (noise as Measured.AtNoiseLevel).threshold, 0.0)
        assertEquals(7.028e-14, noise.value, 0.0)
    }

    /** Граница включительна: ровно порог — достоверная величина. */
    @Test fun measuredIncludesThresholdItself() {
        assertTrue(measured(MACHINE_NOISE_THRESHOLD) is Measured.Reliable)
        assertTrue(measured(-MACHINE_NOISE_THRESHOLD) is Measured.Reliable)
        assertTrue(measured(MACHINE_NOISE_THRESHOLD / 2) is Measured.AtNoiseLevel)
    }

    /** Нефинитные значения не являются измерением: NaN и бесконечности — недостоверны. */
    @Test fun measuredRejectsNonFinite() {
        assertTrue(measured(Double.NaN) is Measured.AtNoiseLevel)
        assertTrue(measured(Double.POSITIVE_INFINITY) is Measured.AtNoiseLevel)
        assertTrue(measured(0.0) is Measured.AtNoiseLevel)
    }

    /** Неположительный порог — ошибка контракта: он означал бы отсутствие проверки. */
    @Test fun measuredRejectsNonPositiveThreshold() {
        assertFailsWith<IllegalArgumentException> { measured(1.0, threshold = 0.0) }
        assertFailsWith<IllegalArgumentException> { measured(1.0, threshold = -1e-13) }
    }

    /** Отношение достоверных величин считается, отношение с участием шума — нет. */
    @Test fun ratioRefusesNoise() {
        assertEquals(2.0, ratio(measured(4e-6), measured(2e-6))!!, 1e-12)
        assertNull(ratio(measured(7.028e-14), measured(3.542e-14)))
        assertNull(ratio(measured(4e-6), measured(3.275e-14)))
        assertNull(ratio(measured(6.539e-14), measured(4e-6)))
    }

    /** Ровно тот случай, из-за которого механизм введён: 7.028e-14 / 3.542e-14 не число. */
    @Test fun realCaseFromTableIsRejected() {
        val printed = listOf(7.028e-14, 3.542e-14, 6.539e-14, 3.275e-14)
        for (v in printed) assertTrue(measured(v) is Measured.AtNoiseLevel, "$v ниже порога")
        assertNull(ratio(measured(printed[0]), measured(printed[1])))
        assertTrue(reliableOrders(printed).all { it == null })
    }

    /** Порядок сходимости по достоверным величинам совпадает с log2 отношения. */
    @Test fun orderOrNullMatchesLog2() {
        assertEquals(2.0, orderOrNull(measured(4e-6), measured(1e-6))!!, 1e-12)
        assertEquals(1.0, orderOrNull(measured(2e-6), measured(1e-6))!!, 1e-12)
    }

    /** Отрицательное отношение логарифма не имеет: порядок не определён. */
    @Test fun orderOrNullRejectsNonPositiveRatio() {
        assertNull(orderOrNull(measured(-4e-6), measured(1e-6)))
    }

    /** Порядок по шуму не считается ни при каком раскладе. */
    @Test fun orderOrNullRejectsNoise() {
        assertNull(orderOrNull(measured(1e-6), measured(3.275e-14)))
        assertNull(orderOrNull(measured(3.275e-14), measured(1e-6)))
    }

    /** На достоверных данных reliableOrders согласован с существующим orders. */
    @Test fun reliableOrdersAgreeWithOrdersOnReliableData() {
        val errs = listOf(1e-3, 2.5e-4, 6.25e-5)
        val ref = orders(errs)
        val rel = reliableOrders(errs)
        assertEquals(errs.size, rel.size)
        assertEquals(ref[0], rel[0]!!, 1e-12)
        assertEquals(ref[1], rel[1]!!, 1e-12)
        assertNull(rel[2]) // последней строке не с чем сравниваться
        assertTrue(ref[2].isNaN())
    }

    /** Смешанная таблица: порядок считается только там, где обе соседние величины достоверны. */
    @Test fun reliableOrdersStopAtNoiseBoundary() {
        val errs = listOf(1e-6, 2.5e-7, 5e-14)
        val rel = reliableOrders(errs)
        assertEquals(2.0, rel[0]!!, 1e-12)
        assertNull(rel[1]) // 5e-14 ниже порога
        assertNull(rel[2])
    }

    /** Порог настраивается: с более мягким порогом те же данные становятся достоверными. */
    @Test fun reliableOrdersRespectCustomThreshold() {
        val errs = listOf(4e-14, 1e-14)
        assertNull(reliableOrders(errs)[0])
        assertEquals(2.0, reliableOrders(errs, threshold = 1e-15)[0]!!, 1e-12)
    }

    /** reliableConstCh: на достоверной величине совпадает с constCh, на шуме — null. */
    @Test fun reliableConstChAgreesWithConstCh() {
        val eh = 0.08
        val h = 0.25
        assertEquals(constCh(eh, h, 2.0), reliableConstCh(measured(eh), h, 2.0)!!, 1e-12)
        assertNull(reliableConstCh(measured(3.542e-14), h, 2.0))
    }
}
