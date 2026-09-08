package numerics

import java.util.concurrent.ForkJoinPool
import java.util.stream.IntStream

/**
 * Параллельная сборка матриц по независимым строкам или столбцам. Каждая задача пишет только
 * в свой участок результата, поэтому результат побитово совпадает с последовательным
 * независимо от числа потоков и порядка выполнения.
 *
 * Перегрузки с `parallel: Boolean` используют общий пул (`ForkJoinPool.commonPool`); перегрузки
 * с [NumericsContext] берут разрешение параллелизма и число потоков из контекста и создают
 * отдельный пул на время сборки.
 */
object ParallelAssembly {

    /**
     * Матрица `rows × cols` как массив строк: строка `i` целиком вычисляется функцией [rowFn]
     * (длина каждой строки должна равняться `cols`).
     */
    fun assembleRows(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        rowFn: (Int) -> DoubleArray,
    ): Array<DoubleArray> = assembleRowsImpl(rows, cols, parallel, Runtime.getRuntime().availableProcessors(), rowFn)

    /** То же, что [assembleRows], с параметрами параллелизма из [context]. */
    fun assembleRows(
        rows: Int,
        cols: Int,
        context: NumericsContext,
        rowFn: (Int) -> DoubleArray,
    ): Array<DoubleArray> = assembleRowsImpl(rows, cols, context.parallel, context.parallelism, rowFn)

    /** Матрица `rows × cols` как массив строк: элемент `(i, j)` вычисляется функцией [cellFn]. */
    fun assembleMatrix(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        cellFn: (Int, Int) -> Double,
    ): Array<DoubleArray> = assembleMatrixImpl(rows, cols, parallel, Runtime.getRuntime().availableProcessors(), cellFn)

    /** То же, что [assembleMatrix], с параметрами параллелизма из [context]. */
    fun assembleMatrix(
        rows: Int,
        cols: Int,
        context: NumericsContext,
        cellFn: (Int, Int) -> Double,
    ): Array<DoubleArray> = assembleMatrixImpl(rows, cols, context.parallel, context.parallelism, cellFn)

    /**
     * Плотная матрица `rows × cols` в столбцовом порядке: элемент `(i, j)` вычисляется функцией
     * [cellFn]; параллелизм — по столбцам.
     */
    fun assembleDense(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        cellFn: (Int, Int) -> Double,
    ): DenseMatrix = assembleDenseImpl(rows, cols, parallel, Runtime.getRuntime().availableProcessors(), cellFn)

    /** То же, что [assembleDense], с параметрами параллелизма из [context]. */
    fun assembleDense(
        rows: Int,
        cols: Int,
        context: NumericsContext,
        cellFn: (Int, Int) -> Double,
    ): DenseMatrix = assembleDenseImpl(rows, cols, context.parallel, context.parallelism, cellFn)

    private fun requireShape(rows: Int, cols: Int) {
        require(rows >= 0 && cols >= 0) { "размеры не могут быть отрицательными: $rows×$cols" }
    }

    private fun assembleRowsImpl(
        rows: Int,
        cols: Int,
        parallel: Boolean,
        parallelism: Int,
        rowFn: (Int) -> DoubleArray,
    ): Array<DoubleArray> {
        requireShape(rows, cols)
        val result = arrayOfNulls<DoubleArray>(rows)
        forEachIndex(rows, parallel, parallelism) { i ->
            val row = rowFn(i)
            require(row.size == cols) { "assembleRows: строка $i длины ${row.size}, ожидалось $cols" }
            result[i] = row
        }
        // Каждый индекс записан ровно один раз (иначе сработал бы require выше), null-ов не остаётся.
        @Suppress("UNCHECKED_CAST")
        return result as Array<DoubleArray>
    }

    private fun assembleMatrixImpl(
        rows: Int,
        cols: Int,
        parallel: Boolean,
        parallelism: Int,
        cellFn: (Int, Int) -> Double,
    ): Array<DoubleArray> {
        requireShape(rows, cols)
        val result = Array(rows) { DoubleArray(cols) }
        forEachIndex(rows, parallel, parallelism) { i ->
            val row = result[i]
            for (j in 0 until cols) row[j] = cellFn(i, j)
        }
        return result
    }

    private fun assembleDenseImpl(
        rows: Int,
        cols: Int,
        parallel: Boolean,
        parallelism: Int,
        cellFn: (Int, Int) -> Double,
    ): DenseMatrix {
        requireShape(rows, cols)
        val data = DoubleArray(rows * cols)
        forEachIndex(cols, parallel, parallelism) { j ->
            val base = j * rows
            for (i in 0 until rows) data[base + i] = cellFn(i, j)
        }
        return DenseMatrix.fromColumnMajor(rows, cols, data)
    }

    /**
     * Выполняет [body] для индексов `0 until count`. Последовательно, если параллелизм запрещён,
     * число потоков равно 1 или индексов меньше двух; иначе — параллельным потоком в общем пуле
     * (`parallelism == availableProcessors`) либо в отдельном пуле заданного размера.
     */
    private fun forEachIndex(count: Int, parallel: Boolean, parallelism: Int, body: (Int) -> Unit) {
        if (!parallel || parallelism == 1 || count < 2) {
            for (i in 0 until count) body(i)
            return
        }
        if (parallelism == Runtime.getRuntime().availableProcessors()) {
            IntStream.range(0, count).parallel().forEach { i -> body(i) }
            return
        }
        val pool = ForkJoinPool(parallelism)
        try {
            pool.submit { IntStream.range(0, count).parallel().forEach { i -> body(i) } }.get()
        } finally {
            pool.shutdown()
        }
    }
}
