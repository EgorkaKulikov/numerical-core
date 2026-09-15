package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Documents that the `parallel` parameter of [ParallelAssembly] methods does not change
 * the assembly result: the sequential and parallel paths produce bit-for-bit
 * identical matrices. This justifies that performance measurements compare the seq/par
 * speedup on the same code.
 *
 * Previously the mode was toggled by the global `ParallelAssembly.parallelEnabled`, and the test
 * had to restore it in `finally`; now the mode is an ordinary argument,
 * so there is nothing to restore and test order does not matter.
 */
@Tag("fast")
class ParallelAssemblyToggleTest {

    /** A non-trivial, non-symmetric cell with different row/column contributions. */
    private fun cellFn(i: Int, j: Int): Double =
        Math.sin(0.3 * i + 1.0) * Math.cos(0.17 * j + 0.5) + (i * 31 + j) % 7

    /** assembleMatrix: parallel=false and =true give bit-for-bit identical results. */
    @Test
    fun assembleMatrix_identical_for_both_modes() {
        val rows = 37
        val cols = 41
        val seq = ParallelAssembly.assembleMatrix(rows, cols, parallel = false, cellFn = ::cellFn)
        val par = ParallelAssembly.assembleMatrix(rows, cols, parallel = true, cellFn = ::cellFn)
        assertTrue(seq.size == par.size, "row count mismatch")
        for (i in 0 until rows) {
            assertTrue(seq[i].contentEquals(par[i]), "row $i differs between seq and par")
        }
    }

    /** assembleRows: parallel=false and =true give bit-for-bit identical results. */
    @Test
    fun assembleRows_identical_for_both_modes() {
        val rows = 37
        val cols = 41
        val rowFn: (Int) -> DoubleArray = { i -> DoubleArray(cols) { j -> cellFn(i, j) } }
        val seq = ParallelAssembly.assembleRows(rows, cols, parallel = false, rowFn = rowFn)
        val par = ParallelAssembly.assembleRows(rows, cols, parallel = true, rowFn = rowFn)
        assertTrue(seq.size == par.size, "row count mismatch")
        for (i in 0 until rows) {
            assertTrue(seq[i].contentEquals(par[i]), "row $i differs between seq and par")
        }
    }
}
