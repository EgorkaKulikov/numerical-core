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
     * Переключатель параллельной сборки (хук для тестов/бенчмарков).
     *
     * По умолчанию `true` — продакшен-поведение не меняется (параллельная
     * сборка через `IntStream.parallel()`). Если выставить `false`, методы
     * [assembleRows]/[assembleMatrix] выполняют ОБЫЧНЫЙ последовательный цикл
     * по тем же [rowFn]/[cellFn]. Это позволяет бенчмарку измерить ускорение
     * seq/par на ОДНОМ И ТОМ ЖЕ коде, не дублируя логику ячеек. Флаг помечен
     * `@Volatile` для безопасной видимости между потоками.
     */
    @Volatile
    var parallelEnabled: Boolean = true

    /**
     * Собирает матрицу [rows] x [cols], вычисляя каждую строку через [rowFn].
     *
     * Для каждого индекса строки `i` вызывается [rowFn], результат которой
     * (длины [cols]) кладётся в строку `i`. Так как каждый индекс пишется ровно
     * одной задачей, операция свободна от гонок данных. При
     * [parallelEnabled]`=true` строки вычисляются параллельно, иначе —
     * последовательным циклом (одинаковый результат).
     */
    fun assembleRows(rows: Int, cols: Int, rowFn: (Int) -> DoubleArray): Array<DoubleArray> {
        val result = Array(rows) { DoubleArray(cols) }
        if (parallelEnabled) {
            IntStream.range(0, rows).parallel().forEach { i -> result[i] = rowFn(i) }
        } else {
            for (i in 0 until rows) result[i] = rowFn(i)
        }
        return result
    }

    /**
     * Собирает матрицу [rows] x [cols] поячеечно: по строкам, последовательно
     * по столбцам внутри строки.
     *
     * Каждая строка обрабатывается одной задачей и заполняется вызовами
     * [cellFn] для всех столбцов; разные задачи пишут в разные строки, поэтому
     * гонок данных нет. При [parallelEnabled]`=true` строки идут параллельно,
     * иначе — последовательным циклом (одинаковый результат).
     */
    fun assembleMatrix(rows: Int, cols: Int, cellFn: (Int, Int) -> Double): Array<DoubleArray> {
        val result = Array(rows) { DoubleArray(cols) }
        val body: (Int) -> Unit = { i ->
            val row = result[i]
            for (j in 0 until cols) row[j] = cellFn(i, j)
        }
        if (parallelEnabled) {
            IntStream.range(0, rows).parallel().forEach { i -> body(i) }
        } else {
            for (i in 0 until rows) body(i)
        }
        return result
    }
}
