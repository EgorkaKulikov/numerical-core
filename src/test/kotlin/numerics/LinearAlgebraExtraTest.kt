package numerics

import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Дополнительные тесты LinearAlgebra: покрывают ветви пропуска нулей в matMat/atWa,
 * транспонированные/векторные операции, нормы и перестановку строк в LU-solve.
 * Эталоны вычисляются из аналитики прямо в тесте — без магических чисел.
 */
class LinearAlgebraExtraTest {
    private val tol = 1e-12

    /** zeros создаёт матрицу нужного размера из нулей. */
    @Test fun zerosShapeAndValues() {
        val z = LinearAlgebra.zeros(2, 3)
        assertEquals(2, z.size); assertEquals(3, z[0].size)
        for (row in z) for (v in row) assertEquals(0.0, v, tol)
    }

    /** matVec: A x для A 2x2 и x — против ручного произведения строк на вектор. */
    @Test fun matVecAgainstManual() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val x = doubleArrayOf(5.0, 6.0)
        val r = LinearAlgebra.matVec(a, x)
        assertEquals(1.0 * 5 + 2.0 * 6, r[0], tol)
        assertEquals(3.0 * 5 + 4.0 * 6, r[1], tol)
    }

    /** matTransVec: A^T y для A 2x2 — сумма столбцов, взвешенных y. */
    @Test fun matTransVecAgainstManual() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val y = doubleArrayOf(1.0, 1.0)
        val r = LinearAlgebra.matTransVec(a, y)
        assertEquals(1.0 + 3.0, r[0], tol) // первый столбец
        assertEquals(2.0 + 4.0, r[1], tol) // второй столбец
    }

    /** matMat: ветвь пропуска нулевого элемента (ail==0.0) при наличии нулей в A. */
    @Test fun matMatSkipsZeroEntries() {
        val a = arrayOf(doubleArrayOf(0.0, 1.0)) // первый множитель нулевой -> continue
        val b = arrayOf(doubleArrayOf(5.0, 5.0), doubleArrayOf(2.0, 3.0))
        val r = LinearAlgebra.matMat(a, b)
        // результат равен второй строке B, т.к. вклад нулевого элемента пропущен
        assertEquals(2.0, r[0][0], tol)
        assertEquals(3.0, r[0][1], tol)
    }

    /** atWa: ветвь пропуска нулевого rwi при нулевом весе строки. */
    @Test fun atWaSkipsZeroWeight() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val w = doubleArrayOf(0.0, 1.0) // первая строка с нулевым весом пропускается
        val g = LinearAlgebra.atWa(a, w)
        // остаётся только вклад второй строки: [3,4]^T [3,4]
        assertEquals(9.0, g[0][0], tol)
        assertEquals(12.0, g[0][1], tol)
        assertEquals(16.0, g[1][1], tol)
        assertTrue(LinearAlgebra.maxAsymmetry(g) < tol)
    }

    /** addScaled: A + s*B поэлементно. */
    @Test fun addScaledElementwise() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0))
        val b = arrayOf(doubleArrayOf(3.0, 4.0))
        val r = LinearAlgebra.addScaled(a, b, 2.0)
        assertEquals(1.0 + 2 * 3.0, r[0][0], tol)
        assertEquals(2.0 + 2 * 4.0, r[0][1], tol)
        // вход не изменился
        assertEquals(1.0, a[0][0], tol)
    }

    /** norm2: евклидова норма (3,4) = 5. */
    @Test fun norm2Pythagorean() {
        assertEquals(5.0, LinearAlgebra.norm2(doubleArrayOf(3.0, 4.0)), tol)
        assertEquals(sqrt(2.0), LinearAlgebra.norm2(doubleArrayOf(1.0, 1.0)), tol)
    }

    /** normInf: максимум модулей координат. */
    @Test fun normInfMaxAbs() {
        assertEquals(7.0, LinearAlgebra.normInf(doubleArrayOf(-3.0, 2.0, -7.0)), tol)
    }

    /** maxAsymmetry: для несимметричной матрицы равен max|A-A^T|. */
    @Test fun maxAsymmetryValue() {
        val a = arrayOf(doubleArrayOf(0.0, 5.0), doubleArrayOf(2.0, 0.0))
        assertEquals(3.0, LinearAlgebra.maxAsymmetry(a), tol) // |5-2|
    }

    /** solve: ветвь перестановки строк (нулевой ведущий элемент в столбце 0). */
    @Test fun solveWithRowSwap() {
        val a = arrayOf(doubleArrayOf(0.0, 2.0), doubleArrayOf(1.0, 1.0))
        val b = doubleArrayOf(2.0, 3.0)
        val x = LinearAlgebra.solve(a, b)
        // проверка подстановкой: 0*x0+2*x1=2 -> x1=1; x0+x1=3 -> x0=2
        assertEquals(2.0, x[0], 1e-10)
        assertEquals(1.0, x[1], 1e-10)
    }

    /** solve: обратный ход на верхнетреугольной 3x3 системе. */
    @Test fun solveBackSubstitution() {
        val a = arrayOf(
            doubleArrayOf(2.0, 1.0, 1.0),
            doubleArrayOf(0.0, 3.0, 1.0),
            doubleArrayOf(0.0, 0.0, 4.0),
        )
        val b = doubleArrayOf(2.0 + 2.0 + 3.0, 6.0 + 3.0, 12.0) // решение (1,2,3)
        val x = LinearAlgebra.solve(a, b)
        assertEquals(1.0, x[0], 1e-10)
        assertEquals(2.0, x[1], 1e-10)
        assertEquals(3.0, x[2], 1e-10)
    }

    /** Регресс #6: пустой/несогласованный вход в публичные методы -> понятное исключение. */
    @Test fun emptyAndMismatchedInputsThrow() {
        val empty = emptyArray<DoubleArray>()
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        // matVec
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.matVec(empty, doubleArrayOf()) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.matVec(a, doubleArrayOf(1.0)) }
        // matMat
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.matMat(empty, a) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.matMat(a, arrayOf(doubleArrayOf(1.0))) }
        // atWa
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.atWa(empty, doubleArrayOf()) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.atWa(a, doubleArrayOf(1.0)) }
        // solve
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.solve(empty, doubleArrayOf()) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.solve(a, doubleArrayOf(1.0)) }
    }

    /** Регресс #6: валидные операции продолжают работать после добавления guard. */
    @Test fun validOperationsStillWork() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val r = LinearAlgebra.matVec(a, doubleArrayOf(1.0, 1.0))
        assertEquals(3.0, r[0], tol)
        assertEquals(7.0, r[1], tol)
    }
}
