package numerics

import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.math.cos
import kotlin.math.sin
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

    /** Один DynamicTest на форму матрицы. */
    private fun perSize(body: (rows: Int, cols: Int) -> Unit): List<DynamicTest> =
        sizes.map { sz -> DynamicTest.dynamicTest("${sz[0]}×${sz[1]}") { body(sz[0], sz[1]) } }

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
    @TestFactory
    fun assembleMatrixEqualsSequential(): List<DynamicTest> = perSize { rows, cols ->
        val expected = sequential(rows, cols)
        val actual = ParallelAssembly.assembleMatrix(rows, cols) { i, j -> cell(i, j, cols) }
        assertEq(expected, actual)
    }

    /** assembleRows равен последовательному заполнению на всех размерах. */
    @TestFactory
    fun assembleRowsEqualsSequential(): List<DynamicTest> = perSize { rows, cols ->
        val expected = sequential(rows, cols)
        val actual = ParallelAssembly.assembleRows(rows, cols) { i ->
            DoubleArray(cols) { j -> cell(i, j, cols) }
        }
        assertEq(expected, actual)
    }

    /**
     * assembleRows отбраковывает строку неверной длины — иначе собралась бы рваная
     * «матрица», и ошибка всплыла бы позже в линейной алгебре. Проверяем оба режима:
     * в параллельном IntStream прокидывает исключение задачи вызывающему потоку как есть.
     */
    @TestFactory
    fun assembleRowsRejectsRaggedRow(): List<DynamicTest> = listOf(false, true).map { parallel ->
        DynamicTest.dynamicTest("parallel=$parallel") {
            val e = assertFailsWith<IllegalArgumentException>("parallel=$parallel") {
                // строка 3 на один элемент короче обявленного cols
                ParallelAssembly.assembleRows(8, 5, parallel) { i ->
                    DoubleArray(if (i == 3) 4 else 5) { j -> cell(i, j, 5) }
                }
            }
            assertTrue(
                e.message!!.contains("assembleRows"),
                "parallel=$parallel: ожидалось сообщение assembleRows, получено ${e.message}",
            )
        }
    }
}
