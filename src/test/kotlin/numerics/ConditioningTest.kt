package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Тесты обусловленности: оценка cond через явное обращение (с невязкой обращения
 * как признаком достоверности) и спектральные оценки методом вращений Якоби.
 *
 * Ключевое требование, которое проверяют тесты: на почти вырожденной матрице API
 * НЕ возвращает молча мусор, а помечает оценку как недостоверную.
 */
@Tag("fast")
class ConditioningTest {

    /** ‖A‖∞ = max по строкам суммы модулей. */
    @Test fun matrixNormInfIsMaxRowSum() {
        val a = arrayOf(doubleArrayOf(1.0, -2.0), doubleArrayOf(3.0, 4.0))
        assertEquals(7.0, Conditioning.matrixNormInf(a), 1e-15)
    }

    /** Пустая матрица — ошибка контракта, а не 0.0. */
    @Test fun matrixNormInfRejectsEmpty() {
        assertFailsWith<IllegalArgumentException> { Conditioning.matrixNormInf(arrayOf()) }
        assertFailsWith<IllegalArgumentException> { Conditioning.matrixNormInf(arrayOf(doubleArrayOf())) }
    }

    /** Обращение 2x2: сверка с явной формулой обратной матрицы. */
    @Test fun inverseMatchesClosedForm() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val inv = Conditioning.inverse(a)!!
        // A^{-1} = 1/det * [[d, -b], [-c, a]], det = -2
        assertEquals(-2.0, inv[0][0], 1e-12)
        assertEquals(1.0, inv[0][1], 1e-12)
        assertEquals(1.5, inv[1][0], 1e-12)
        assertEquals(-0.5, inv[1][1], 1e-12)
    }

    /** Неквадратная и пустая матрицы отвергаются контрактом обращения. */
    @Test fun inverseRejectsMalformed() {
        assertFailsWith<IllegalArgumentException> { Conditioning.inverse(arrayOf()) }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.inverse(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))
        }
    }

    /** cond∞ для 2x2 совпадает с аналитическим значением ‖A‖∞·‖A^{-1}‖∞ = 7*3 = 21. */
    @Test fun conditionInfMatchesAnalyticValue() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val est = Conditioning.conditionInf(a)
        assertTrue(est.isReliable, "хорошо обусловленная матрица обязана давать достоверную оценку")
        assertEquals(21.0, est.condInf, 1e-10)
        assertEquals(21.0, est.valueOrNull()!!, 1e-10)
        assertTrue(est.inversionResidual < 1e-14, "невязка обращения = ${est.inversionResidual}")
    }

    /** cond∞(I) = 1 — нижняя граница числа обусловленности достигается. */
    @Test fun conditionInfOfIdentityIsOne() {
        val est = Conditioning.conditionInf(LinearAlgebra.identity(5))
        assertTrue(est.isReliable)
        assertEquals(1.0, est.condInf, 1e-12)
    }

    /** Вырожденная матрица: оценка бесконечна, помечена недостоверной, valueOrNull = null. */
    @Test fun conditionInfReportsSingularAsUnreliable() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0))
        val est = Conditioning.conditionInf(a)
        assertTrue(!est.isReliable, "вырожденная матрица не может дать достоверную оценку")
        assertNull(est.valueOrNull())
        assertNull(Conditioning.inverse(a))
    }

    /** Неположительный порог достоверности — ошибка контракта: он означал бы отсутствие проверки. */
    @Test fun conditionInfRejectsNonPositiveTolerance() {
        assertFailsWith<IllegalArgumentException> {
            Conditioning.conditionInf(LinearAlgebra.identity(2), tolerance = 0.0)
        }
    }

    /** Собственные значения симметричной 2x2 [[2,1],[1,2]] — это 1 и 3, по возрастанию. */
    @Test fun symmetricEigenvaluesOfTwoByTwo() {
        val a = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 2.0))
        val eig = Conditioning.symmetricEigenvalues(a)
        assertEquals(1.0, eig[0], 1e-12)
        assertEquals(3.0, eig[1], 1e-12)
        assertEquals(1.0, Conditioning.smallestMagnitudeEigenvalue(a), 1e-12)
        assertEquals(3.0, Conditioning.conditionSymmetric(a), 1e-12)
    }

    /** Диагональная матрица уже диагональна: метод обязан выйти сразу и не испортить спектр. */
    @Test fun symmetricEigenvaluesOfDiagonalAreDiagonalEntries() {
        val a = arrayOf(
            doubleArrayOf(3.0, 0.0, 0.0),
            doubleArrayOf(0.0, -1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 2.0),
        )
        val eig = Conditioning.symmetricEigenvalues(a)
        assertEquals(-1.0, eig[0], 1e-14)
        assertEquals(2.0, eig[1], 1e-14)
        assertEquals(3.0, eig[2], 1e-14)
        // min|lambda| = 1, max|lambda| = 3
        assertEquals(3.0, Conditioning.conditionSymmetric(a), 1e-12)
    }

    /** Нулевые внедиагональные элементы пропускаются без вращения: 3x3 со связью только (0,2). */
    @Test fun symmetricEigenvaluesSkipsZeroOffDiagonal() {
        val a = arrayOf(
            doubleArrayOf(2.0, 0.0, 1.0),
            doubleArrayOf(0.0, 5.0, 0.0),
            doubleArrayOf(1.0, 0.0, 2.0),
        )
        val eig = Conditioning.symmetricEigenvalues(a)
        assertEquals(1.0, eig[0], 1e-12)
        assertEquals(3.0, eig[1], 1e-12)
        assertEquals(5.0, eig[2], 1e-12)
    }

    /** Сумма собственных значений равна следу — независимая проверка на несимметричном спектре. */
    @Test fun symmetricEigenvaluesPreserveTrace() {
        val n = 6
        val a = Array(n) { i -> DoubleArray(n) { j -> 1.0 / (1.0 + i + j) } }
        val eig = Conditioning.symmetricEigenvalues(a)
        var trace = 0.0
        for (i in 0 until n) trace += a[i][i]
        assertEquals(trace, eig.sum(), 1e-10)
    }

    /** Несимметричная матрица, пустая, неквадратная и maxSweeps < 1 отвергаются контрактом. */
    @Test fun symmetricEigenvaluesRejectMalformed() {
        assertFailsWith<IllegalArgumentException> {
            Conditioning.symmetricEigenvalues(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 1.0)))
        }
        assertFailsWith<IllegalArgumentException> { Conditioning.symmetricEigenvalues(arrayOf()) }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.symmetricEigenvalues(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(1.0, 2.0, 3.0)))
        }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.symmetricEigenvalues(LinearAlgebra.identity(2), maxSweeps = 0)
        }
    }

    /** Симметричная вырожденная матрица: cond2 = +Inf, а не большое случайное число. */
    @Test fun conditionSymmetricOfSingularIsInfinite() {
        val a = arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 1.0))
        assertEquals(0.0, Conditioning.smallestMagnitudeEigenvalue(a), 1e-14)
        assertTrue(Conditioning.conditionSymmetric(a).isInfinite())
    }

    /**
     * ГРАНИЦА ПРИМЕНИМОСТИ, часть 1 (достоверный режим): на хорошо обусловленной
     * симметричной матрице оценка через обращение и метод Якоби согласованы.
     *
     * Для симметричной A нормы ‖·‖∞ и ‖·‖2 различаются не более чем в n раз,
     * поэтому сверяем не сами числа, а то, что обе оценки лежат в одном коридоре
     * (и обе конечны, и обе >= 1).
     */
    @Test fun wellConditionedMatrixAgreesBetweenMethods() {
        val n = 16
        // A = I + 0.1 * симметричное гладкое ядро -> диагонально доминирующая, cond ~ 1.
        val a = Array(n) { i -> DoubleArray(n) { j -> (if (i == j) 1.0 else 0.0) + 0.1 / (1.0 + abs(i - j)) } }
        val est = Conditioning.conditionInf(a)
        val cond2 = Conditioning.conditionSymmetric(a)
        assertTrue(est.isReliable, "невязка обращения = ${est.inversionResidual}")
        assertTrue(cond2.isFinite() && cond2 >= 1.0)
        assertTrue(est.condInf >= 1.0)
        assertTrue(est.condInf <= n * cond2 && cond2 <= n * est.condInf, "cond_inf=${est.condInf}, cond_2=$cond2")
    }

    /**
     * ГРАНИЦА ПРИМЕНИМОСТИ, часть 2 (недостоверный режим): матрица дискретизации
     * ядра 1/(1+t+s) численно вырождена.
     *
     * Метод Якоби честно даёт sigma_min = 0 (эталонное наблюдение Round 21), тогда
     * как оценка через обращение при этом ЛИБО признаётся недостоверной по невязке,
     * ЛИБО бесконечна. Именно этот контракт закрывает класс ошибок «напечатали шум
     * порядка 1e16...1e19 как число обусловленности».
     */
    @Test fun nearlySingularKernelMatrixIsReportedUnreliable() {
        val n = 32
        val a = Array(n) { i ->
            DoubleArray(n) { j -> 1.0 / (1.0 + i.toDouble() / n + j.toDouble() / n) }
        }
        val sigmaMin = Conditioning.smallestMagnitudeEigenvalue(a)
        assertTrue(sigmaMin < 1e-14 * Conditioning.matrixNormInf(a), "sigma_min = $sigmaMin")
        assertTrue(Conditioning.conditionSymmetric(a) > 1e14)
        val est = Conditioning.conditionInf(a)
        assertTrue(
            !est.isReliable,
            "оценка через обращение обязана быть помечена недостоверной: cond=${est.condInf}, невязка=${est.inversionResidual}",
        )
        assertNull(est.valueOrNull())
    }

    /**
     * Регуляризация выводит ту же матрицу в достоверный режим: при alpha = 1e-2
     * невязка обращения падает ниже порога и оценка становится пригодной.
     */
    @Test fun regularizationRestoresReliability() {
        val n = 32
        val a = Array(n) { i ->
            DoubleArray(n) { j ->
                (if (i == j) 1e-2 else 0.0) + 1.0 / (1.0 + i.toDouble() / n + j.toDouble() / n)
            }
        }
        val est = Conditioning.conditionInf(a)
        assertTrue(est.isReliable, "невязка обращения = ${est.inversionResidual}")
        assertTrue(est.condInf > 1.0 && est.condInf.isFinite())
    }
}
