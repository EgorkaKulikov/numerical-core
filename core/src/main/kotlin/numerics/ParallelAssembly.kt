package numerics

import java.util.concurrent.ForkJoinPool
import java.util.stream.IntStream

/**
 * Parallel matrix assembly over independent rows or columns. Each task writes only to
 * its own slice of the result, so the result is bit-for-bit identical to the sequential one
 * regardless of the number of threads and execution order.
 *
 * The `parallel: Boolean` overloads use the common pool (`ForkJoinPool.commonPool`); the
 * [NumericsContext] overloads take the parallelism flag and thread count from the context and
 * create a dedicated pool for the duration of the assembly.
 */
public object ParallelAssembly {

    /**
     * Matrix `rows × cols` as an array of rows: row `i` is computed as a whole by [rowFn]
     * (each row must have length `cols`).
     */
    public fun assembleRows(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        rowFn: (Int) -> DoubleArray,
    ): Array<DoubleArray> = assembleRowsImpl(rows, cols, parallel, Runtime.getRuntime().availableProcessors(), rowFn)

    /** Same as [assembleRows], with parallelism settings taken from [context]. */
    public fun assembleRows(
        rows: Int,
        cols: Int,
        context: NumericsContext,
        rowFn: (Int) -> DoubleArray,
    ): Array<DoubleArray> = assembleRowsImpl(rows, cols, context.parallel, context.parallelism, rowFn)

    /** Matrix `rows × cols` as an array of rows: entry `(i, j)` is computed by [cellFn]. */
    public fun assembleMatrix(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        cellFn: (Int, Int) -> Double,
    ): Array<DoubleArray> = assembleMatrixImpl(rows, cols, parallel, Runtime.getRuntime().availableProcessors(), cellFn)

    /** Same as [assembleMatrix], with parallelism settings taken from [context]. */
    public fun assembleMatrix(
        rows: Int,
        cols: Int,
        context: NumericsContext,
        cellFn: (Int, Int) -> Double,
    ): Array<DoubleArray> = assembleMatrixImpl(rows, cols, context.parallel, context.parallelism, cellFn)

    /**
     * Dense matrix `rows × cols` in column-major order: entry `(i, j)` is computed by
     * [cellFn]; parallelized over columns.
     */
    public fun assembleDense(
        rows: Int,
        cols: Int,
        parallel: Boolean = true,
        cellFn: (Int, Int) -> Double,
    ): DenseMatrix = assembleDenseImpl(rows, cols, parallel, Runtime.getRuntime().availableProcessors(), cellFn)

    /** Same as [assembleDense], with parallelism settings taken from [context]. */
    public fun assembleDense(
        rows: Int,
        cols: Int,
        context: NumericsContext,
        cellFn: (Int, Int) -> Double,
    ): DenseMatrix = assembleDenseImpl(rows, cols, context.parallel, context.parallelism, cellFn)

    private fun requireShape(rows: Int, cols: Int) {
        require(rows >= 0 && cols >= 0) { "Matrix dimensions must be non-negative: $rows×$cols" }
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
            require(row.size == cols) { "assembleRows: row $i has length ${row.size}, expected $cols" }
            result[i] = row
        }
        // Every index is written exactly once (otherwise the require above would have fired), so no nulls remain.
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
     * Runs [body] for indices `0 until count`. Sequentially if parallelism is disabled,
     * the thread count is 1, or there are fewer than two indices; otherwise as a parallel stream
     * in the common pool (`parallelism == availableProcessors`) or in a dedicated pool of the given size.
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
