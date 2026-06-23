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
     * Собирает матрицу [rows] x [cols], параллельно вычисляя каждую строку.
     *
     * Для каждого индекса строки `i` вызывается [rowFn], результат которой
     * (длины [cols]) кладётся в строку `i`. Так как каждый индекс пишется ровно
     * одной задачей, операция свободна от гонок данных.
     */
    fun assembleRows(rows: Int, cols: Int, rowFn: (Int) -> DoubleArray): Array<DoubleArray> {
        val result = Array(rows) { DoubleArray(cols) }
        IntStream.range(0, rows).parallel().forEach { i -> result[i] = rowFn(i) }
        return result
    }

    /**
     * Собирает матрицу [rows] x [cols] поячеечно: параллельно по строкам,
     * последовательно по столбцам внутри строки.
     *
     * Каждая строка обрабатывается одной задачей и заполняется вызовами
     * [cellFn] для всех столбцов; разные задачи пишут в разные строки, поэтому
     * гонок данных нет.
     */
    fun assembleMatrix(rows: Int, cols: Int, cellFn: (Int, Int) -> Double): Array<DoubleArray> {
        val result = Array(rows) { DoubleArray(cols) }
        IntStream.range(0, rows).parallel().forEach { i ->
            val row = result[i]
            for (j in 0 until cols) row[j] = cellFn(i, j)
        }
        return result
    }
}
