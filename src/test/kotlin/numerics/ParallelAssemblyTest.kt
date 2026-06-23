package numerics

import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Проверка, что параллельная сборка матриц совпадает с последовательной.
 *
 * Параллельная сборка по независимым строкам обязана давать побитово тот же
 * результат, что и последовательное заполнение: используется крупный размер
 * (200x200), чтобы реально задействовать пул потоков ForkJoin.
 */
class ParallelAssemblyTest {

    private val sizes = arrayOf(
        intArrayOf(1, 1),
        intArrayOf(3, 5),
        intArrayOf(17, 4),
        intArrayOf(200, 200),
    )

    private fun cell(i: Int, j: Int, cols: Int): Double = sin(i.toDouble()) * cos(j.toDouble()) + i.toDouble() * cols + j

    private fun sequential(rows: Int, cols: Int): Array<DoubleArray> =
        Array(rows) { i -> DoubleArray(cols) { j -> cell(i, j, cols) } }

    private fun assertEq(expected: Array<DoubleArray>, actual: Array<DoubleArray>) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            assertEquals(expected[i].size, actual[i].size)
            for (j in expected[i].indices) {
                assertTrue(expected[i][j] == actual[i][j], "mismatch at [$i][$j]")
            }
        }
    }

    /** assembleMatrix равен последовательному заполнению на всех размерах. */
    @Test
    fun assembleMatrixEqualsSequential() {
        for (sz in sizes) {
            val (rows, cols) = sz[0] to sz[1]
            val expected = sequential(rows, cols)
            val actual = ParallelAssembly.assembleMatrix(rows, cols) { i, j -> cell(i, j, cols) }
            assertEq(expected, actual)
        }
    }

    /** assembleRows равен последовательному заполнению на всех размерах. */
    @Test
    fun assembleRowsEqualsSequential() {
        for (sz in sizes) {
            val (rows, cols) = sz[0] to sz[1]
            val expected = sequential(rows, cols)
            val actual = ParallelAssembly.assembleRows(rows, cols) { i ->
                DoubleArray(cols) { j -> cell(i, j, cols) }
            }
            assertEq(expected, actual)
        }
    }
}
