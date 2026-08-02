package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Документирует, что параметр `parallel` методов [ParallelAssembly] НЕ меняет
 * результат сборки: последовательный и параллельный пути дают ПОБИТОВО
 * идентичные матрицы. Это обоснование того, что бенчмарк меряет seq/par
 * ускорение на одном коде.
 *
 * Раньше режим переключался глобальным `ParallelAssembly.parallelEnabled`, и тест
 * обязан был восстанавливать его в `finally`; теперь режим — обычный аргумент,
 * поэтому восстанавливать нечего и порядок тестов ни на что не влияет.
 */
@Tag("fast")
class ParallelAssemblyToggleTest {

    /** Нетривиальная, не-симметричная ячейка с разным вкладом строки/столбца. */
    private fun cellFn(i: Int, j: Int): Double =
        Math.sin(0.3 * i + 1.0) * Math.cos(0.17 * j + 0.5) + (i * 31 + j) % 7

    /** assembleMatrix: parallel=false и =true дают побитно одинаковый результат. */
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

    /** assembleRows: parallel=false и =true дают побитно одинаковый результат. */
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
