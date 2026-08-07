package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Тесты диагностики ПРЯМОЙ ошибки решения СЛАУ.
 *
 * Проверяемое требование: малая ОБРАТНАЯ ошибка (единственное, что контролирует
 * постпроверка `solve` по порогу [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE])
 * не должна выглядеть как признак точности результата. На численно вырожденном
 * входе API обязан либо честно сказать «границы нет», либо пометить оценку
 * недостоверной, но НЕ возвращать число, выглядящее как правда.
 */
@Tag("fast")
class ForwardErrorTest {

    // --- Обратная ошибка: измерение, а не оценка -----------------------------

    /** На точном решении невязка нулевая, значит и относительная обратная ошибка равна 0. */
    @Test fun backwardErrorOfExactSolutionIsZero() {
        val a = arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 4.0))
        val b = doubleArrayOf(2.0, 4.0)
        assertEquals(0.0, Conditioning.relativeBackwardError(a, b, doubleArrayOf(1.0, 1.0)), 0.0)
    }

    /**
     * Значение совпадает с формулой `‖Ax−b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)` — той же, что
     * стоит в постпроверке `solve`. Здесь ‖A‖∞ = 4, ‖x‖∞ = 1, ‖b‖∞ = 4,
     * ‖Ax−b‖∞ = |2·1 − 3| = 1, значит ответ 1/4.
     */
    @Test fun backwardErrorMatchesSolveNormalisation() {
        val a = arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 4.0))
        val b = doubleArrayOf(3.0, 4.0)
        assertEquals(0.25, Conditioning.relativeBackwardError(a, b, doubleArrayOf(1.0, 1.0)), 1e-15)
    }

    /** Нулевой масштаб (A = 0, b = 0) даёт 0.0, а не NaN от деления ноль на ноль. */
    @Test fun backwardErrorOfZeroSystemIsZeroNotNaN() {
        val a = arrayOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0, 0.0))
        val z = doubleArrayOf(0.0, 0.0)
        assertEquals(0.0, Conditioning.relativeBackwardError(a, z, z), 0.0)
    }

    /** Пустая, неквадратная и несогласованные по длине входы отвергаются контрактом. */
    @Test fun backwardErrorRejectsMalformed() {
        val a = arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0))
        val v = doubleArrayOf(1.0, 1.0)
        assertFailsWith<IllegalArgumentException> {
            Conditioning.relativeBackwardError(arrayOf(), DoubleArray(0), DoubleArray(0))
        }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.relativeBackwardError(arrayOf(doubleArrayOf(1.0, 2.0, 3.0)), doubleArrayOf(1.0), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException> { Conditioning.relativeBackwardError(a, doubleArrayOf(1.0), v) }
        assertFailsWith<IllegalArgumentException> { Conditioning.relativeBackwardError(a, v, doubleArrayOf(1.0)) }
    }

    // --- Три режима ForwardError --------------------------------------------

    /** Достоверная оценка cond даёт границу `cond · ω` и число уцелевших разрядов. */
    @Test fun reliableConditionGivesBoundedForwardError() {
        val est = ConditionEstimate(1e6, 1e-15, Conditioning.INVERSION_RESIDUAL_TOLERANCE)
        val fe = Conditioning.forwardError(est, 1e-16)
        assertTrue(fe is ForwardError.Bounded)
        assertEquals(1e-10, fe.relativeBound, 1e-24)
        assertEquals(1e-10, fe.relativeBoundOrNull()!!, 1e-24)
        assertEquals(1e-16, fe.backwardError, 0.0)
        assertEquals(10.0, fe.survivingDigitsOrNull()!!, 1e-12)
    }

    /**
     * НЕДОСТОВЕРНАЯ оценка cond НЕ превращается в число: режим [ForwardError.Unreliable],
     * `relativeBoundOrNull() == null`. Это и есть защита от «числа, выглядящего как правда».
     */
    @Test fun unreliableConditionYieldsNoNumber() {
        val est = ConditionEstimate(3.7e18, 760.0, Conditioning.INVERSION_RESIDUAL_TOLERANCE)
        val fe = Conditioning.forwardError(est, 1e-16)
        assertTrue(fe is ForwardError.Unreliable)
        assertNull(fe.relativeBoundOrNull())
        assertNull(fe.survivingDigitsOrNull())
        // Сама недостоверная оценка сохранена для печати с пометкой и разбора причины.
        assertEquals(760.0, fe.condition.inversionResidual, 0.0)
    }

    /**
     * «Конечного cond нет вовсе» отличается от «cond большой, но конечный»:
     * бесконечная оценка даёт [ForwardError.NoFiniteBound], а не [ForwardError.Unreliable].
     */
    @Test fun infiniteConditionIsDistinctFromUnreliable() {
        val est = ConditionEstimate(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1e-8)
        val fe = Conditioning.forwardError(est, 1e-16)
        assertTrue(fe is ForwardError.NoFiniteBound)
        assertNull(fe.relativeBoundOrNull())
        assertEquals(1e-16, fe.backwardError, 0.0)
    }

    /** Нулевая обратная ошибка означает полную мантиссу: 16 разрядов, а не бесконечность. */
    @Test fun zeroBackwardErrorGivesFullMantissa() {
        val est = ConditionEstimate(21.0, 0.0, 1e-8)
        val fe = Conditioning.forwardError(est, 0.0)
        assertEquals(0.0, fe.relativeBoundOrNull()!!, 0.0)
        assertEquals(16.0, fe.survivingDigitsOrNull()!!, 0.0)
    }

    /** Граница >= 1 означает «ни одного гарантированно верного разряда», а не отрицательное число. */
    @Test fun boundAboveOneMeansZeroSurvivingDigits() {
        val est = ConditionEstimate(1e14, 1e-15, 1e-8)
        val fe = Conditioning.forwardError(est, 1e-13)
        assertTrue(fe.relativeBoundOrNull()!! > 1.0)
        assertEquals(0.0, fe.survivingDigitsOrNull()!!, 0.0)
    }

    /** Нефинитная или отрицательная обратная ошибка — ошибка контракта: это не измерение. */
    @Test fun forwardErrorRejectsNonMeasuredBackwardError() {
        val est = ConditionEstimate(21.0, 0.0, 1e-8)
        assertFailsWith<IllegalArgumentException> { Conditioning.forwardError(est, Double.NaN) }
        assertFailsWith<IllegalArgumentException> { Conditioning.forwardError(est, -1e-16) }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.forwardErrorSymmetric(LinearAlgebra.identity(2), Double.POSITIVE_INFINITY)
        }
    }

    // --- Спектральный путь: sigma_min = 0 отличим от «cond велико, но конечно» ---

    /**
     * ТОЧНО вырожденная симметричная матрица: спектральный путь честно говорит
     * «конечной границы нет», а не выдаёт большое случайное число.
     */
    @Test fun symmetricPathReportsNoFiniteBoundOnSingularMatrix() {
        val a = arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 1.0))
        assertEquals(0.0, Conditioning.smallestMagnitudeEigenvalue(a), 1e-14)
        val fe = Conditioning.forwardErrorSymmetric(a, 1e-16)
        assertTrue(fe is ForwardError.NoFiniteBound)
        assertNull(fe.relativeBoundOrNull())
    }

    /** На хорошо обусловленной симметричной матрице спектральный путь даёт конечную границу. */
    @Test fun symmetricPathBoundsWellConditionedMatrix() {
        val a = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 2.0))
        val fe = Conditioning.forwardErrorSymmetric(a, 1e-16)
        assertTrue(fe is ForwardError.Bounded)
        // cond2 = 3/1 = 3, значит граница = 3e-16.
        assertEquals(3.0, fe.cond, 1e-12)
        assertEquals(3e-16, fe.relativeBound, 1e-28)
    }

    /** Несимметричная матрица на спектральном пути — ошибка контракта, а не тихий неверный ответ. */
    @Test fun symmetricPathRejectsAsymmetricMatrix() {
        assertFailsWith<IllegalArgumentException> {
            Conditioning.forwardErrorSymmetric(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 1.0)), 1e-16)
        }
    }

    // --- solveDiagnosed ------------------------------------------------------

    /**
     * Решение из [LinearAlgebra.solveDiagnosed] ПОБИТОВО совпадает с решением
     * [LinearAlgebra.solve]: диагностика ничего не уточняет и не меняет.
     */
    @Test fun diagnosedSolutionIsBitwiseIdenticalToSolve() {
        val n = 16
        val a = Array(n) { i -> DoubleArray(n) { j -> (if (i == j) 1.0 else 0.0) + 0.1 / (1.0 + abs(i - j)) } }
        val b = DoubleArray(n) { 1.0 }
        val plain = LinearAlgebra.solve(a, b)
        val diagnosed = LinearAlgebra.solveDiagnosed(a, b)
        for (i in 0 until n) {
            assertEquals(plain[i].toRawBits(), diagnosed.x[i].toRawBits(), "компонента $i разошлась")
        }
    }

    /**
     * Хорошо обусловленная система: граница прямой ошибки конечна и мала,
     * значащих разрядов сохраняется почти вся мантисса.
     *
     * Эталон зафиксирован текущим прогоном: `cond∞ ≈ 1.669`, граница ~5.1e-16,
     * то есть больше пятнадцати верных десятичных разрядов.
     */
    @Test fun wellConditionedSystemKeepsAlmostAllDigits() {
        val n = 16
        val a = Array(n) { i -> DoubleArray(n) { j -> (if (i == j) 1.0 else 0.0) + 0.1 / (1.0 + abs(i - j)) } }
        val b = DoubleArray(n) { 1.0 }
        val fe = LinearAlgebra.solveDiagnosed(a, b).forwardError
        assertTrue(fe is ForwardError.Bounded, "хорошо обусловленная система обязана давать границу, получено $fe")
        assertTrue(fe.cond < 2.0, "cond=${fe.cond}")
        assertTrue(fe.relativeBound < 1e-14, "граница=${fe.relativeBound}")
        assertTrue(fe.survivingDigitsOrNull()!! > 14.0, "разрядов=${fe.survivingDigitsOrNull()}")
    }

    /**
     * КЛЮЧЕВОЙ СЛУЧАЙ ИЗ ISSUE #10. Матрица дискретизации ядра `1/(1 + t + s)`
     * численно вырождена: обратная ошибка решения остаётся на уровне 1e-18 —
     * постпроверка `solve` молчит и решение возвращается как ни в чём не бывало, —
     * тогда как прямая ошибка НЕ ограничена ничем осмысленным.
     *
     * Диагностика обязана это показать, причём ОБА пути обязаны отказаться выдать
     * маленькое число: путь через обращение помечает оценку недостоверной
     * (измеренная невязка обращения — сотни), спектральный путь даёт границу,
     * заведомо превышающую единицу, то есть «ни одного верного разряда».
     */
    @Test fun nearlySingularSystemHidesLossOfAccuracyFromSolve() {
        val n = 32
        val a = Array(n) { i -> DoubleArray(n) { j -> 1.0 / (1.0 + i.toDouble() / n + j.toDouble() / n) } }
        val b = DoubleArray(n) { 1.0 }

        // Обратная ошибка мала — постпроверка solve пропускает решение без единого сигнала.
        val x = LinearAlgebra.solve(a, b)
        val omega = Conditioning.relativeBackwardError(a, b, x)
        assertTrue(omega < 1e-14, "обратная ошибка обязана остаться малой, получено $omega")

        // Путь через обращение: числа нет, и это сказано явно.
        val viaInversion = LinearAlgebra.solveDiagnosed(a, b).forwardError
        assertTrue(viaInversion is ForwardError.Unreliable, "получено $viaInversion")
        assertNull(viaInversion.relativeBoundOrNull())
        assertNull(viaInversion.survivingDigitsOrNull())
        assertTrue(
            viaInversion.condition.inversionResidual > Conditioning.INVERSION_RESIDUAL_TOLERANCE,
            "невязка обращения=${viaInversion.condition.inversionResidual}",
        )

        // Спектральный путь: граница существует, но она больше единицы — верных разрядов нет.
        val viaSpectrum = LinearAlgebra
            .solveDiagnosed(a, b, source = LinearAlgebra.ConditionSource.SYMMETRIC_SPECTRUM)
            .forwardError
        assertTrue(viaSpectrum is ForwardError.Bounded, "получено $viaSpectrum")
        assertTrue(viaSpectrum.relativeBound > 1.0, "граница=${viaSpectrum.relativeBound}")
        assertEquals(0.0, viaSpectrum.survivingDigitsOrNull()!!, 0.0)
    }

    /**
     * Контракт вырожденности НЕ ослаблен: на точно вырожденной матрице
     * [LinearAlgebra.solveDiagnosed] бросает исключение так же, как [LinearAlgebra.solve],
     * и до диагностики дело не доходит.
     */
    @Test fun diagnosedSolveKeepsSingularityContract() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0))
        val b = doubleArrayOf(1.0, 3.0)
        assertFailsWith<IllegalStateException> { LinearAlgebra.solve(a, b) }
        assertFailsWith<IllegalStateException> { LinearAlgebra.solveDiagnosed(a, b) }
    }

    /** Порог достоверности пробрасывается: заниженный порог делает достоверную оценку недостоверной. */
    @Test fun toleranceIsForwardedToConditionEstimate() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val b = doubleArrayOf(1.0, 1.0)
        assertTrue(LinearAlgebra.solveDiagnosed(a, b).forwardError is ForwardError.Bounded)
        val strict = LinearAlgebra.solveDiagnosed(a, b, tolerance = Double.MIN_VALUE).forwardError
        assertTrue(strict is ForwardError.Unreliable, "получено $strict")
    }
}
