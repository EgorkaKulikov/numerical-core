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
 * Checks that parallel matrix assembly matches the sequential one.
 *
 * Parallel assembly over independent rows must produce bit-for-bit the same
 * result as sequential filling: a large size (200x200) is used
 * so that the ForkJoin thread pool is actually engaged.
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

    /** One DynamicTest per matrix shape. */
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

    /** assembleMatrix equals sequential filling for all sizes. */
    @TestFactory
    fun assembleMatrixEqualsSequential(): List<DynamicTest> = perSize { rows, cols ->
        val expected = sequential(rows, cols)
        val actual = ParallelAssembly.assembleMatrix(rows, cols) { i, j -> cell(i, j, cols) }
        assertEq(expected, actual)
    }

    /** assembleRows equals sequential filling for all sizes. */
    @TestFactory
    fun assembleRowsEqualsSequential(): List<DynamicTest> = perSize { rows, cols ->
        val expected = sequential(rows, cols)
        val actual = ParallelAssembly.assembleRows(rows, cols) { i ->
            DoubleArray(cols) { j -> cell(i, j, cols) }
        }
        assertEq(expected, actual)
    }

    /**
     * assembleRows rejects a row of the wrong length — otherwise a ragged "matrix"
     * would be assembled and the error would surface later in linear algebra. Both modes are checked:
     * in parallel mode IntStream rethrows the task exception to the calling thread as is.
     */
    @TestFactory
    fun assembleRowsRejectsRaggedRow(): List<DynamicTest> = listOf(false, true).map { parallel ->
        DynamicTest.dynamicTest("parallel=$parallel") {
            val e = assertFailsWith<IllegalArgumentException>("parallel=$parallel") {
                // row 3 is one element shorter than the declared cols
                ParallelAssembly.assembleRows(8, 5, parallel) { i ->
                    DoubleArray(if (i == 3) 4 else 5) { j -> cell(i, j, 5) }
                }
            }
            assertTrue(
                e.message!!.contains("assembleRows"),
                "parallel=$parallel: expected an assembleRows message, got ${e.message}",
            )
        }
    }
}
