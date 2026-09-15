package numerics

/**
 * Dense matrix backed by a flat column-major array: element (i, j) is stored at
 * `data[i + j * rows]`. This layout matches the BLAS/LAPACK convention and lets
 * the data be passed to native libraries without repacking.
 *
 * The shape is fixed at construction; the contents are mutable — [set] writes
 * directly into [data]. Instances are created via the companion-object factories.
 *
 * @property rows number of rows.
 * @property cols number of columns.
 * @property data storage of length `rows * cols` in column-major order.
 */
public class DenseMatrix private constructor(
    public val rows: Int,
    public val cols: Int,
    public val data: DoubleArray,
) {
    init {
        require(rows >= 0 && cols >= 0) { "Matrix dimensions must be non-negative: $rows x $cols" }
        require(data.size == rows * cols) { "Array length ${data.size} does not equal rows*cols = ${rows * cols}" }
    }

    /** True if the number of rows equals the number of columns. */
    public val isSquare: Boolean get() = rows == cols

    private fun checkIndex(i: Int, j: Int) {
        require(i in 0 until rows && j in 0 until cols) { "Index ($i, $j) is out of bounds for matrix $rows x $cols" }
    }

    /** Returns element (i, j); throws [IllegalArgumentException] if the index is out of bounds. */
    public operator fun get(i: Int, j: Int): Double {
        checkIndex(i, j)
        return data[i + j * rows]
    }

    /** Stores [v] into element (i, j); throws [IllegalArgumentException] if the index is out of bounds. */
    public operator fun set(i: Int, j: Int, v: Double) {
        checkIndex(i, j)
        data[i + j * rows] = v
    }

    /** Returns an independent copy of the matrix (the data array is copied). */
    public fun copy(): DenseMatrix = DenseMatrix(rows, cols, data.copyOf())

    /** Returns a new cols x rows matrix — a transposed copy. */
    public fun transpose(): DenseMatrix {
        val t = DoubleArray(rows * cols)
        for (j in 0 until cols) {
            val base = j * rows
            for (i in 0 until rows) t[j + i * cols] = data[base + i]
        }
        return DenseMatrix(cols, rows, t)
    }

    /** Returns a copy of row [i] of length [cols]; throws [IllegalArgumentException] for an invalid index. */
    public fun row(i: Int): DoubleArray {
        require(i in 0 until rows) { "Row index $i is out of bounds for matrix $rows x $cols" }
        return DoubleArray(cols) { j -> data[i + j * rows] }
    }

    /** Returns a copy of column [j] of length [rows]; throws [IllegalArgumentException] for an invalid index. */
    public fun column(j: Int): DoubleArray {
        require(j in 0 until cols) { "Column index $j is out of bounds for matrix $rows x $cols" }
        return data.copyOfRange(j * rows, (j + 1) * rows)
    }

    /** Returns a row-major copy of the matrix: an array of [rows] rows, each of length [cols]. */
    public fun toRows(): Array<DoubleArray> = Array(rows) { i -> DoubleArray(cols) { j -> data[i + j * rows] } }

    /** Matrices are equal if their shapes and all elements match (bitwise, as in [DoubleArray.contentEquals]). */
    override fun equals(other: Any?): Boolean =
        other is DenseMatrix && rows == other.rows && cols == other.cols && data.contentEquals(other.data)

    /** Hash code consistent with [equals]: based on shape and contents. */
    override fun hashCode(): Int = 31 * (31 * rows + cols) + data.contentHashCode()

    /** Shape and at most the top-left 6 x 6 elements; truncation is marked with an ellipsis. */
    override fun toString(): String {
        val sb = StringBuilder("DenseMatrix($rows x $cols)")
        val r = minOf(rows, PREVIEW)
        val c = minOf(cols, PREVIEW)
        for (i in 0 until r) {
            sb.append('\n')
            for (j in 0 until c) {
                if (j > 0) sb.append(' ')
                sb.append(data[i + j * rows])
            }
            if (cols > PREVIEW) sb.append(" …")
        }
        if (rows > PREVIEW) sb.append("\n…")
        return sb.toString()
    }

    /** Matrix factories: zero, identity, diagonal, from rows, from a column-major array, from a function. */
    public companion object {
        private const val PREVIEW = 6

        /** Creates a zero matrix rows x cols; throws [IllegalArgumentException] for negative dimensions. */
        public fun zeros(rows: Int, cols: Int): DenseMatrix {
            require(rows >= 0 && cols >= 0) { "Matrix dimensions must be non-negative: $rows x $cols" }
            return DenseMatrix(rows, cols, DoubleArray(rows * cols))
        }

        /** Creates the n x n identity matrix. */
        public fun identity(n: Int): DenseMatrix {
            val m = zeros(n, n)
            for (i in 0 until n) m.data[i + i * n] = 1.0
            return m
        }

        /**
         * Builds a matrix from an array of rows, copying the data. An empty array yields a 0 x 0 matrix.
         * Throws [IllegalArgumentException] if the rows have different lengths.
         */
        public fun fromRows(rows: Array<DoubleArray>): DenseMatrix {
            if (rows.isEmpty()) return DenseMatrix(0, 0, DoubleArray(0))
            val n = rows.size
            val cols = rows[0].size
            for (i in rows.indices) {
                require(rows[i].size == cols) { "Row $i has length ${rows[i].size}, expected $cols" }
            }
            val data = DoubleArray(n * cols)
            for (i in 0 until n) {
                val r = rows[i]
                for (j in 0 until cols) data[i + j * n] = r[j]
            }
            return DenseMatrix(n, cols, data)
        }

        /**
         * Wraps an existing column-major array without copying: the matrix takes ownership of
         * the array, and any later changes to the array are visible through the matrix (and vice versa).
         * Throws [IllegalArgumentException] if the dimensions are negative or the array length is not `rows * cols`.
         */
        public fun fromColumnMajor(rows: Int, cols: Int, data: DoubleArray): DenseMatrix = DenseMatrix(rows, cols, data)

        /** Builds a rows x cols matrix, computing each element (i, j) with [cell]. */
        public fun build(rows: Int, cols: Int, cell: (Int, Int) -> Double): DenseMatrix {
            val m = zeros(rows, cols)
            val data = m.data
            for (j in 0 until cols) {
                val base = j * rows
                for (i in 0 until rows) data[base + i] = cell(i, j)
            }
            return m
        }

        /** Creates an n x n diagonal matrix with [d] on the diagonal, where n is the length of [d]. */
        public fun diagonal(d: DoubleArray): DenseMatrix {
            val n = d.size
            val m = zeros(n, n)
            for (i in 0 until n) m.data[i + i * n] = d[i]
            return m
        }
    }
}
