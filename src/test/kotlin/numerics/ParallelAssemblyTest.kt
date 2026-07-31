package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Проверка, что параллельная сборка матриц совпадает с последовательной.
 *
 * Параллельная сборка по независимым строкам обязана давать побитово тот же
 * результат, что и последовательное заполнение: используется крупный размер
 * (200x200), чтобы реально задействовать пул потоков ForkJoin.
 */
@Tag("fast")
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

    /**
     * assembleRows отбраковывает строку неверной длины — иначе собралась бы рваная
     * «матрица», и ошибка всплыла бы позже в линейной алгебре. Проверяем ОБА режима:
     * в параллельном IntStream прокидывает исключение задачи вызывающему потоку как есть.
     */
    @Test
    fun assembleRowsRejectsRaggedRow() {
        val saved = ParallelAssembly.parallelEnabled
        try {
            for (parallel in listOf(false, true)) {
                ParallelAssembly.parallelEnabled = parallel
                val e = assertFailsWith<IllegalArgumentException>("parallel=$parallel") {
                    // строка 3 на один элемент короче обявленного cols
                    ParallelAssembly.assembleRows(8, 5) { i ->
                        DoubleArray(if (i == 3) 4 else 5) { j -> cell(i, j, 5) }
                    }
                }
                assertTrue(
                    e.message!!.contains("assembleRows"),
                    "parallel=$parallel: ожидалось сообщение assembleRows, получено ${e.message}",
                )
            }
        } finally {
            ParallelAssembly.parallelEnabled = saved
        }
    }
}
