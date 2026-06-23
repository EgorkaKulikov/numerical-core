package numerics

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Перекрёстная проверка multik/OpenBLAS-бэкенда [LinearAlgebra] против
 * чистой ручной реализации [ReferenceLinearAlgebra].
 *
 * На псевдослучайных, но фиксированных по сидам данных оба бэкенда обязаны
 * совпадать с точностью до 1e-8: нативный OpenBLAS и ручной LU не должны
 * расходиться на хорошо обусловленных входах.
 */
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
}
