package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Перекрёстная проверка реализаций [numerics.backend.Backends.java] и (при доступности)
 * [numerics.backend.Backends.native] через единую точку входа [LinearAlgebra] против независимого
 * оракула [ReferenceOracle] на фиксированных по сидам данных с допуском 1e-8.
 */
@Tag("fast")
@Suppress("DEPRECATION")
class LinearAlgebraVsReferenceTest {

    private val tol = 1e-8
    private val sizes = intArrayOf(3, 8, 20)
    private val backends: List<LinAlgBackend> = Backends.available()

    /** Один DynamicTest на пару (реализация, размер): падение одного случая не скрывает остальные. */
    private fun perBackendAndSize(name: String, body: (LinAlgBackend, Int) -> Unit): List<DynamicTest> =
        backends.flatMap { backend ->
            sizes.map { n -> DynamicTest.dynamicTest("$name [${backend.name}, n=$n]") { body(backend, n) } }
        }

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
    @TestFactory
    fun matVecMatchesReference() = perBackendAndSize("matVecMatchesReference") { backend, n ->
        val rnd = Random(1000 + n)
        val a = randMatrix(rnd, n, n)
        val x = randVector(rnd, n)
        assertVecEq(ReferenceOracle.matVec(a, x), LinearAlgebra.matVec(a, x, backend))
    }

    /** matTransVec бэкенда совпадает с эталоном (прямоугольные матрицы). */
    @TestFactory
    fun matTransVecMatchesReference() = perBackendAndSize("matTransVecMatchesReference") { backend, n ->
        val rnd = Random(2000 + n)
        val a = randMatrix(rnd, n, n + 2)
        val y = randVector(rnd, n)
        assertVecEq(ReferenceOracle.matTransVec(a, y), LinearAlgebra.matTransVec(a, y, backend))
    }

    /** matMat бэкенда совпадает с эталоном на прямоугольных множителях. */
    @TestFactory
    fun matMatMatchesReference() = perBackendAndSize("matMatMatchesReference") { backend, n ->
        val rnd = Random(3000 + n)
        val a = randMatrix(rnd, n, n + 1)
        val b = randMatrix(rnd, n + 1, n + 3)
        assertMatEq(ReferenceOracle.matMat(a, b), LinearAlgebra.matMat(a, b, backend))
    }

    /** atWa (A^T diag(w) A) бэкенда совпадает с эталоном. */
    @TestFactory
    fun atWaMatchesReference() = perBackendAndSize("atWaMatchesReference") { backend, n ->
        val rnd = Random(4000 + n)
        val a = randMatrix(rnd, n + 2, n)
        val w = randVector(rnd, n + 2)
        assertMatEq(ReferenceOracle.atWa(a, w), LinearAlgebra.atWa(a, w, backend))
    }

    /** addScaled (A + s*B) бэкенда совпадает с эталоном. */
    @TestFactory
    fun addScaledMatchesReference() = perBackendAndSize("addScaledMatchesReference") { backend, n ->
        val rnd = Random(5000 + n)
        val a = randMatrix(rnd, n, n)
        val b = randMatrix(rnd, n, n)
        val s = rnd.nextDouble(-2.0, 2.0)
        assertMatEq(ReferenceOracle.addScaled(a, b, s), LinearAlgebra.addScaled(a, b, s, backend))
    }

    /** solve бэкенда совпадает с эталоном на хорошо обусловленных СЛАУ. */
    @TestFactory
    fun solveMatchesReference() = perBackendAndSize("solveMatchesReference") { backend, n ->
        val rnd = Random(6000 + n)
        val a = diagDominant(rnd, n)
        val b = randVector(rnd, n)
        assertVecEq(ReferenceOracle.solve(a, b), LinearAlgebra.solve(a, b, backend))
    }

    /**
     * Единая семантика вырожденности: на точно вырожденной СЛАУ любого масштаба
     * оба бэкенда обязаны бросить исключение одного типа через единую точку входа.
     *
     * Прогоняется на всех доступных реализациях (`-Dnumerics.backend=native|java`).
     *
     * О выборе случая. Здесь именно точно вырожденные матрицы (ранг < n),
     * а не плохо обусловленные (типа матрицы Гильберта): плохая обусловленность
     * сама по себе не является вырожденностью — плохо обусловленные, но
     * невырожденные матрицы (cond ~ 1e10) решаются с малой невязкой, и требовать
     * от бэкендов одинаковой реакции на них было бы неверно (см. KDoc
     * [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE]).
     */
    @TestFactory
    fun bothBackendsRejectSingularSystemsAtAnyScale(): List<DynamicTest> {
        val cases = mutableListOf<DynamicTest>()
        // (а) Ранг 1 в разных масштабах — масштабно-инвариантность семантики.
        for (scale in doubleArrayOf(1e-8, 1.0, 1e8)) {
            cases += DynamicTest.dynamicTest("ранг 1, scale=$scale") {
                val a = arrayOf(
                    doubleArrayOf(1.0 * scale, 2.0 * scale),
                    doubleArrayOf(2.0 * scale, 4.0 * scale),
                )
                val b = doubleArrayOf(1.0 * scale, 3.0 * scale)
                assertFailsWith<IllegalStateException>("scale=$scale, бэкенд ${Backends.default().name}") {
                    LinearAlgebra.solve(a, b)
                }
            }
        }
        cases += DynamicTest.dynamicTest("вырождение поворотом и ранг 2 в размере 3") {
        // (б) Вырождение поворотом: diag(1, 0) в базисе, повёрнутом на 45 градусов —
        // ни один элемент матрицы не мал, а сама она вырождена.
        val rotated = arrayOf(doubleArrayOf(0.5, 0.5), doubleArrayOf(0.5, 0.5))
        assertFailsWith<IllegalStateException>("бэкенд ${Backends.default().name}") {
            LinearAlgebra.solve(rotated, doubleArrayOf(1.0, 0.0))
        }
        // (в) Вырождение в большем размере: вторая строка — ровно удвоенная первая.
        // Все числа — степени двойки, поэтому вырожденность точная в IEEE-754 и её
        // видит любой LU (ведущий элемент обращается в ровно ноль).
        val rank2 = arrayOf(
            doubleArrayOf(1.0, 2.0, 4.0),
            doubleArrayOf(2.0, 4.0, 8.0),
            doubleArrayOf(1.0, 4.0, 16.0),
        )
        assertFailsWith<IllegalStateException>("бэкенд ${Backends.default().name}") {
            LinearAlgebra.solve(rank2, doubleArrayOf(1.0, 3.0, 1.0))
        }
        }
        return cases
    }

    /**
     * Граница контракта, выявленная фактически и зафиксированная здесь как
     * осознанная: критерий по невязке не ловит несовместную почти вырожденную
     * систему, если бэкенд вернул решение очень большой нормы.
     *
     * Причина принципиальная, а не дефект порога: при `||x|| ~ 1e16` невязка
     * порядка `||b||` всё равно мала относительно `||A||*||x||`, то есть такой x
     * точно решает близкую систему (малая обратная ошибка) — формально честный
     * ответ. Отбраковывать его можно только проверкой обусловленности, которая
     * запрещена требованием нейтральности (плохо обусловленные, но невырожденные
     * матрицы с cond ~ 1e10 обязаны решаться). Тест закрепляет именно то, что
     * гарантируется: любой бэкенд либо бросает [IllegalStateException], либо возвращает
     * числовое решение с малой обратной ошибкой — но никогда не возвращает NaN/Inf
     * и не врёт о невязке.
     */
    @Test
    fun inconsistentRankDeficientSystemEitherThrowsOrHasSmallBackwardError() {
        // Третья строка = сумма двух первых, а правая часть — нет (4 != 1+2).
        val a = arrayOf(
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
            doubleArrayOf(5.0, 7.0, 9.0),
        )
        val b = doubleArrayOf(1.0, 2.0, 4.0)
        val x: DoubleArray? = try {
            LinearAlgebra.solve(a, b)
        } catch (e: IllegalStateException) {
            // Честный отказ — тоже допустимый исход контракта (так ведёт себя ручной LU оракула),
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
            // именно это и проверила единая точка входа, пропустив его. Норма ||A||_inf считается по самой
            // матрице, а не литералом: завышенная константа сделала бы границу слабее
            // той, что уже проверена в единой точке входа, и тест не ловил бы ничего нового.
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
     * Плохо обусловленная, но невырожденная СЛАУ (матрица Гильберта 8x8,
     * cond ~ 1e10) не отбраковывается порогом невязки в единой точке входа.
     *
     * Это защита от регрессии: плохо обусловленные, но невырожденные матрицы
     * с такой обусловленностью обязаны решаться, и ужесточение порога отбраковало
     * бы их вместо нечислового результата.
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
