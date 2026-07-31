package numerics

import numerics.backend.Backends
import org.junit.jupiter.api.Tag
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Перекрёстная проверка multik/OpenBLAS-бэкенда [LinearAlgebra] против
 * чистой ручной реализации [ReferenceLinearAlgebra].
 *
 * На псевдослучайных, но фиксированных по сидам данных оба бэкенда обязаны
 * совпадать с точностью до 1e-8: нативный OpenBLAS и ручной LU не должны
 * расходиться на хорошо обусловленных входах.
 */
@Tag("fast")
class LinearAlgebraVsReferenceTest {

    private val tol = 1e-8
    private val sizes = intArrayOf(3, 8, 20)

    private fun randMatrix(rnd: Random, rows: Int, cols: Int): Array<DoubleArray> =
        Array(rows) { DoubleArray(cols) { rnd.nextDouble(-1.0, 1.0) } }

    private fun randVector(rnd: Random, n: Int): DoubleArray =
        DoubleArray(n) { rnd.nextDouble(-1.0, 1.0) }

    /** Диагонально доминирующая (значит невырожденная) матрица n x n. */
    private fun diagDominant(rnd: Random, n: Int): Array<DoubleArray> {
        val a = randMatrix(rnd, n, n)
        for (i in 0 until n) {
            var rowSum = 0.0
            for (j in 0 until n) rowSum += kotlin.math.abs(a[i][j])
            a[i][i] += rowSum + 1.0
        }
        return a
    }

    private fun assertMatEq(expected: Array<DoubleArray>, actual: Array<DoubleArray>) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            assertEquals(expected[i].size, actual[i].size)
            for (j in expected[i].indices) {
                assertTrue(
                    kotlin.math.abs(expected[i][j] - actual[i][j]) < tol,
                    "mismatch at [$i][$j]: ${expected[i][j]} vs ${actual[i][j]}"
                )
            }
        }
    }

    private fun assertVecEq(expected: DoubleArray, actual: DoubleArray) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            assertTrue(
                kotlin.math.abs(expected[i] - actual[i]) < tol,
                "mismatch at [$i]: ${expected[i]} vs ${actual[i]}"
            )
        }
    }

    /** matVec бэкенда совпадает с эталоном на размерах 3, 8, 20. */
    @Test
    fun matVecMatchesReference() {
        for (n in sizes) {
            val rnd = Random(1000 + n)
            val a = randMatrix(rnd, n, n)
            val x = randVector(rnd, n)
            assertVecEq(ReferenceLinearAlgebra.matVec(a, x), LinearAlgebra.matVec(a, x))
        }
    }

    /** matTransVec бэкенда совпадает с эталоном (прямоугольные матрицы). */
    @Test
    fun matTransVecMatchesReference() {
        for (n in sizes) {
            val rnd = Random(2000 + n)
            val a = randMatrix(rnd, n, n + 2)
            val y = randVector(rnd, n)
            assertVecEq(ReferenceLinearAlgebra.matTransVec(a, y), LinearAlgebra.matTransVec(a, y))
        }
    }

    /** matMat бэкенда совпадает с эталоном на прямоугольных множителях. */
    @Test
    fun matMatMatchesReference() {
        for (n in sizes) {
            val rnd = Random(3000 + n)
            val a = randMatrix(rnd, n, n + 1)
            val b = randMatrix(rnd, n + 1, n + 3)
            assertMatEq(ReferenceLinearAlgebra.matMat(a, b), LinearAlgebra.matMat(a, b))
        }
    }

    /** atWa (A^T diag(w) A) бэкенда совпадает с эталоном. */
    @Test
    fun atWaMatchesReference() {
        for (n in sizes) {
            val rnd = Random(4000 + n)
            val a = randMatrix(rnd, n + 2, n)
            val w = randVector(rnd, n + 2)
            assertMatEq(ReferenceLinearAlgebra.atWa(a, w), LinearAlgebra.atWa(a, w))
        }
    }

    /** addScaled (A + s*B) бэкенда совпадает с эталоном. */
    @Test
    fun addScaledMatchesReference() {
        for (n in sizes) {
            val rnd = Random(5000 + n)
            val a = randMatrix(rnd, n, n)
            val b = randMatrix(rnd, n, n)
            val s = rnd.nextDouble(-2.0, 2.0)
            assertMatEq(ReferenceLinearAlgebra.addScaled(a, b, s), LinearAlgebra.addScaled(a, b, s))
        }
    }

    /** solve бэкенда совпадает с эталоном на хорошо обусловленных СЛАУ. */
    @Test
    fun solveMatchesReference() {
        for (n in sizes) {
            val rnd = Random(6000 + n)
            val a = diagDominant(rnd, n)
            val b = randVector(rnd, n)
            assertVecEq(ReferenceLinearAlgebra.solve(a, b), LinearAlgebra.solve(a, b))
        }
    }

    /**
     * Единая семантика вырожденности: на ТОЧНО вырожденной СЛАУ любого масштаба
     * ОБА бэкенда обязаны бросить исключение ОДНОГО ТИПА через фасад.
     *
     * До переноса проверки в фасад это было неверно: `MultikCpuBackend.solve`
     * контролировал только NaN/Inf, а LAPACK в части случаев возвращает конечный
     * мусор. Прогонять обязательно НА ОБОИХ бэкендах
     * (`-Dnumerics.backend=multik` и `-Dnumerics.backend=reference`).
     *
     * О ВЫБОРЕ КЕЙСА. Здесь именно ТОЧНО вырожденные матрицы (ранг < n),
     * а не плохо ОБУСЛОВЛЕННЫЕ (типа матрицы Гильберта): плохая обусловленность
     * сама по себе НЕ является вырожденностью — штатные задачи первого рода
     * с регуляризацией дают cond ~ 1e10 и решаются с малой невязкой, и требовать
     * от бэкендов одинаковой реакции на них было бы неверно (см. KDoc
     * [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE]).
     */
    @Test
    fun bothBackendsRejectSingularSystemsAtAnyScale() {
        // (а) Ранг 1 в разных масштабах — масштабно-инвариантность семантики.
        for (scale in doubleArrayOf(1e-8, 1.0, 1e8)) {
            val a = arrayOf(
                doubleArrayOf(1.0 * scale, 2.0 * scale),
                doubleArrayOf(2.0 * scale, 4.0 * scale),
            )
            val b = doubleArrayOf(1.0 * scale, 3.0 * scale)
            assertFailsWith<IllegalStateException>("scale=$scale, активен ${Backends.active.name}") {
                LinearAlgebra.solve(a, b)
            }
        }
        // (б) Вырождение поворотом: diag(1, 0) в базисе, повёрнутом на 45 градусов —
        // ни один элемент матрицы не мал, а сама она вырождена.
        val rotated = arrayOf(doubleArrayOf(0.5, 0.5), doubleArrayOf(0.5, 0.5))
        assertFailsWith<IllegalStateException>("активен ${Backends.active.name}") {
            LinearAlgebra.solve(rotated, doubleArrayOf(1.0, 0.0))
        }
        // (в) Вырождение в большем размере: вторая строка — РОВНО удвоенная первая.
        // Все числа — степени двойки, поэтому вырожденность ТОЧНАЯ в IEEE-754 и её
        // видит любой LU (ведущий элемент обращается в ровно ноль).
        val rank2 = arrayOf(
            doubleArrayOf(1.0, 2.0, 4.0),
            doubleArrayOf(2.0, 4.0, 8.0),
            doubleArrayOf(1.0, 4.0, 16.0),
        )
        assertFailsWith<IllegalStateException>("активен ${Backends.active.name}") {
            LinearAlgebra.solve(rank2, doubleArrayOf(1.0, 3.0, 1.0))
        }
    }

    /**
     * ГРАНИЦА КОНТРАКТА, выявленная фактически и зафиксированная здесь как
     * ОСОЗНАННАЯ: критерий по невязке НЕ ловит НЕСОВМЕСТНУЮ почти вырожденную
     * систему, если бэкенд вернул решение ОГРОМНОЙ нормы.
     *
     * Причина принципиальная, а не дефект порога: при `||x|| ~ 1e16` невязка
     * порядка `||b||` всё равно мала относительно `||A||*||x||`, то есть такой x
     * точно решает БЛИЗКУЮ систему (малая ОБРАТНАЯ ошибка) — формально честный
     * ответ. Отбраковывать его можно только проверкой ОБУСЛОВЛЕННОСТИ, которая
     * запрещена требованием нейтральности (штатные задачи первого рода
     * имеют cond ~ 1e10 и обязаны решаться). Тест закрепляет именно то, что
     * гарантируется: любой бэкенд либо бросает [IllegalStateException], либо возвращает
     * числовое решение с малой обратной ошибкой — но НИКОГДА не возвращает NaN/Inf
     * и не врёт о невязке.
     */
    @Test
    fun inconsistentRankDeficientSystemEitherThrowsOrHasSmallBackwardError() {
        // Третья строка = сумма двух первых, а правая часть — НЕТ (4 != 1+2).
        val a = arrayOf(
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
            doubleArrayOf(5.0, 7.0, 9.0),
        )
        val b = doubleArrayOf(1.0, 2.0, 4.0)
        val x: DoubleArray? = try {
            LinearAlgebra.solve(a, b)
        } catch (e: IllegalStateException) {
            // Честный отказ — тоже допустимый исход контракта (так ведёт себя reference),
            // но и он обязан быть содержательным: сообщение называет причину.
            // Без этой проверки тест был бы тавтологией на бэкенде reference.
            assertTrue(
                e.message?.contains("вырожден") == true,
                "отказ обязан называть причину: ${e.message}",
            )
            null
        }
        if (x != null) {
            for (v in x) assertTrue(v.isFinite(), "NaN/Inf запрещён контрактом, получено $v")
            // Если решение всё-таки вернулось, оно обязано иметь малую обратную ошибку:
            // именно это и проверил фасад, пропустив его. Норма ||A||_inf считается по САМОЙ
            // матрице, а не литералом: завышенная константа сделала бы границу слабее
            // той, что уже проверил фасад, и тест не ловил бы ничего нового.
            val matrixNormInf = (0..2).maxOf { i -> (0..2).sumOf { j -> kotlin.math.abs(a[i][j]) } }
            val residual = LinearAlgebra.normInf(
                DoubleArray(3) { i -> (0..2).sumOf { j -> a[i][j] * x[j] } - b[i] }
            )
            val bound = LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE *
                maxOf(matrixNormInf * LinearAlgebra.normInf(x), LinearAlgebra.normInf(b))
            assertTrue(residual <= bound, "невязка $residual обязана быть <= $bound")
        }
    }

    /**
     * Плохо обусловленная, но НЕВЫРОЖДЕННАЯ СЛАУ (матрица Гильберта 8x8,
     * cond ~ 1e10) НЕ отбраковывается порогом невязки в фасаде.
     *
     * Это защита от регрессии: реальные задачи первого рода с регуляризацией
     * имеют ровно такую обусловленность, и ужесточение порога сломало бы их
     * вместо отлова мусора.
     */
    @Test
    fun illConditionedButSolvableSystemIsAccepted() {
        val n = 8
        val hilbert = Array(n) { i -> DoubleArray(n) { j -> 1.0 / (i + j + 1) } }
        val rhs = DoubleArray(n) { 1.0 }
        val x = LinearAlgebra.solve(hilbert, rhs)
        // Фактическая проверка того, что решение вообще вернулось и числовое.
        assertEquals(n, x.size)
        for (v in x) assertTrue(v.isFinite(), "решение обязано быть числовым, получено $v")
    }
}
