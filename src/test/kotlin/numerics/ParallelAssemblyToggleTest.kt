package numerics

import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Документирует, что переключатель [ParallelAssembly.parallelEnabled] НЕ меняет
 * результат сборки: последовательный и параллельный пути дают идентичные матрицы.
 * Это обоснование того, что бенчмарк меряет seq/par ускорение на одном коде.
 */
class ParallelAssemblyToggleTest {

    /** Нетривиальная, не-симметричная ячейка с разным вкладом строки/столбца. */
    private fun cellFn(i: Int, j: Int): Double =
        Math.sin(0.3 * i + 1.0) * Math.cos(0.17 * j + 0.5) + (i * 31 + j) % 7

    /** assembleMatrix: parallelEnabled=false и =true дают побитно одинаковый результат. */
    @Test
    fun assembleMatrix_identical_for_both_modes() {
        val rows = 37
        val cols = 41
        val saved = ParallelAssembly.parallelEnabled
        try {
            ParallelAssembly.parallelEnabled = false
            val seq = ParallelAssembly.assembleMatrix(rows, cols, ::cellFn)
            ParallelAssembly.parallelEnabled = true
            val par = ParallelAssembly.assembleMatrix(rows, cols, ::cellFn)
            assertTrue(seq.size == par.size, "row count mismatch")
            for (i in 0 until rows) {
                assertTrue(seq[i].contentEquals(par[i]), "row $i differs between seq and par")
            }
        } finally {
            ParallelAssembly.parallelEnabled = saved
        }
    }

    /** assembleRows: parallelEnabled=false и =true дают побитно одинаковый результат. */
    @Test
    fun assembleRows_identical_for_both_modes() {
        val rows = 37
        val cols = 41
        val rowFn: (Int) -> DoubleArray = { i -> DoubleArray(cols) { j -> cellFn(i, j) } }
        val saved = ParallelAssembly.parallelEnabled
        try {
            ParallelAssembly.parallelEnabled = false
            val seq = ParallelAssembly.assembleRows(rows, cols, rowFn)
            ParallelAssembly.parallelEnabled = true
            val par = ParallelAssembly.assembleRows(rows, cols, rowFn)
            assertTrue(seq.size == par.size, "row count mismatch")
            for (i in 0 until rows) {
                assertTrue(seq[i].contentEquals(par[i]), "row $i differs between seq and par")
            }
        } finally {
            ParallelAssembly.parallelEnabled = saved
        }
    }
}
