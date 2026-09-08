package numerics

import java.util.stream.IntStream

/**
 * Помощник для параллельной сборки матриц по независимым строкам.
 *
 * Сборка матрицы (например, матрицы Грама/коллокации) часто распадается на
 * вычисление строк, не зависящих друг от друга. Каждая параллельная задача
 * пишет в СВОЙ индекс строки выходного массива, поэтому гонок данных нет:
 * разные потоки никогда не обращаются к одной и той же ячейке, а публикация
 * результата гарантируется барьером завершения `forEach`/`parallel()` в
 * java.util.stream. Зависимостей, кроме стандартной библиотеки JVM, нет.
 */
object ParallelAssembly {

    /**
     * Собирает матрицу [rows] x [cols], вычисляя каждую строку через [rowFn].
     *
     * Для каждого индекса строки `i` вызывается [rowFn], результат которой
     * (длины [cols]) кладётся в строку `i`. Так как каждый индекс пишется ровно
     * одной задачей, операция свободна от гонок данных. При [parallel]`=true`
     * строки вычисляются параллельно, иначе — ОБЫЧНЫМ последовательным циклом по
     * тому же [rowFn] (побитово одинаковый результат). Режим передаётся
     * ПАРАМЕТРОМ, а не глобальным переключателем: иначе бенчмарк, меряющий
     * seq/par, менял бы поведение чужого кода в той же JVM.
     *
     * Массив-накопитель создаётся ПУСТЫМ ([arrayOfNulls]): строки приходят готовыми из
     * [rowFn], поэтому предварительное `Array(rows) { DoubleArray(cols) }` выделяло бы
     * rows*cols чисел лишь для того, чтобы тут же их выбросить присваиванием — а это
     * горячий путь сборки матриц M/M2. Длина каждой строки проверяется: иначе «матрица»
     * могла бы молча получиться рваной, и ошибка проявилась бы много позже, в линейной
     * алгебре, уже без связи с причиной.
     *
     * @throws IllegalArgumentException если [rowFn] вернула строку длины, отличной от [cols].
     */
    fun assembleRows(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        rowFn: (Int) -> DoubleArray,
    ): Array<DoubleArray> {
        val result = arrayOfNulls<DoubleArray>(rows)
        val body: (Int) -> Unit = { i ->
            val row = rowFn(i)
            require(row.size == cols) { "assembleRows: строка $i длины ${row.size}, ожидалось $cols" }
            result[i] = row
        }
        if (parallel) {
            IntStream.range(0, rows).parallel().forEach { i -> body(i) }
        } else {
            for (i in 0 until rows) body(i)
        }
        // Каждый индекс 0..rows-1 записан ровно один раз (иначе сработал бы require выше и
        // управление сюда не дошло), поэтому null-ов в массиве не остаётся.
        @Suppress("UNCHECKED_CAST")
        return result as Array<DoubleArray>
    }

    /**
     * Собирает матрицу [rows] x [cols] поячеечно: по строкам, последовательно
     * по столбцам внутри строки.
     *
     * Каждая строка обрабатывается одной задачей и заполняется вызовами
     * [cellFn] для всех столбцов; разные задачи пишут в разные строки, поэтому
     * гонок данных нет. При [parallel]`=true` строки идут параллельно, иначе —
     * последовательным циклом (побитово одинаковый результат).
     */
    fun assembleMatrix(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        cellFn: (Int, Int) -> Double,
    ): Array<DoubleArray> {
        val result = Array(rows) { DoubleArray(cols) }
        val body: (Int) -> Unit = { i ->
            val row = result[i]
            for (j in 0 until cols) row[j] = cellFn(i, j)
        }
        if (parallel) {
            IntStream.range(0, rows).parallel().forEach { i -> body(i) }
        } else {
            for (i in 0 until rows) body(i)
        }
        return result
    }

    /**
     * Собирает матрицу [rows] x [cols] в плоском столбцовом формате [DenseMatrix],
     * вычисляя каждую ячейку через [cellFn].
     *
     * Единица параллелизма — столбец: одна задача заполняет непрерывный отрезок
     * `data[j * rows until (j + 1) * rows]`, и разные задачи никогда не касаются
     * одних и тех же ячеек, поэтому гонок данных нет. При [parallel]`=true` столбцы
     * идут параллельно, иначе — последовательным циклом; результат побитово одинаков
     * в обоих режимах и совпадает с [assembleMatrix] после [DenseMatrix.fromRows].
     *
     * @throws IllegalArgumentException при отрицательных размерах.
     */
    fun assembleDense(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        cellFn: (Int, Int) -> Double,
    ): DenseMatrix {
        require(rows >= 0 && cols >= 0) { "assembleDense: размеры не могут быть отрицательными: $rows x $cols" }
        val data = DoubleArray(rows * cols)
        val body: (Int) -> Unit = { j ->
            val base = j * rows
            for (i in 0 until rows) data[base + i] = cellFn(i, j)
        }
        if (parallel) {
            IntStream.range(0, cols).parallel().forEach { j -> body(j) }
        } else {
            for (j in 0 until cols) body(j)
        }
        return DenseMatrix.fromColumnMajor(rows, cols, data)
    }
}
