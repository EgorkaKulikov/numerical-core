package numerics

/**
 * Плотная матрица над плоским массивом в столбцовом порядке (column-major):
 * элемент (i, j) лежит в `data[i + j * rows]`. Такой порядок совпадает с
 * соглашением BLAS/LAPACK и позволяет передавать данные в нативные библиотеки
 * без перекладки.
 *
 * Размеры фиксируются при создании; содержимое изменяемое — [set] пишет прямо
 * в [data]. Экземпляры создаются фабриками из сопутствующего объекта.
 *
 * @property rows число строк.
 * @property cols число столбцов.
 * @property data хранилище длиной `rows * cols` в столбцовом порядке.
 */
class DenseMatrix private constructor(
    val rows: Int,
    val cols: Int,
    val data: DoubleArray,
) {
    init {
        require(rows >= 0 && cols >= 0) { "Размеры матрицы не могут быть отрицательными: $rows x $cols" }
        require(data.size == rows * cols) { "Длина массива ${data.size} не равна rows*cols = ${rows * cols}" }
    }

    /** Истина, если число строк равно числу столбцов. */
    val isSquare: Boolean get() = rows == cols

    private fun checkIndex(i: Int, j: Int) {
        require(i in 0 until rows && j in 0 until cols) { "Индекс ($i, $j) вне матрицы $rows x $cols" }
    }

    /** Возвращает элемент (i, j); бросает [IllegalArgumentException], если индекс вне матрицы. */
    operator fun get(i: Int, j: Int): Double {
        checkIndex(i, j)
        return data[i + j * rows]
    }

    /** Записывает [v] в элемент (i, j); бросает [IllegalArgumentException], если индекс вне матрицы. */
    operator fun set(i: Int, j: Int, v: Double) {
        checkIndex(i, j)
        data[i + j * rows] = v
    }

    /** Возвращает независимую копию матрицы (массив данных копируется). */
    fun copy(): DenseMatrix = DenseMatrix(rows, cols, data.copyOf())

    /** Возвращает новую матрицу cols x rows — транспонированную копию. */
    fun transpose(): DenseMatrix {
        val t = DoubleArray(rows * cols)
        for (j in 0 until cols) {
            val base = j * rows
            for (i in 0 until rows) t[j + i * cols] = data[base + i]
        }
        return DenseMatrix(cols, rows, t)
    }

    /** Возвращает копию строки [i] длиной [cols]; бросает [IllegalArgumentException] при недопустимом индексе. */
    fun row(i: Int): DoubleArray {
        require(i in 0 until rows) { "Индекс строки $i вне матрицы $rows x $cols" }
        return DoubleArray(cols) { j -> data[i + j * rows] }
    }

    /** Возвращает копию столбца [j] длиной [rows]; бросает [IllegalArgumentException] при недопустимом индексе. */
    fun column(j: Int): DoubleArray {
        require(j in 0 until cols) { "Индекс столбца $j вне матрицы $rows x $cols" }
        return data.copyOfRange(j * rows, (j + 1) * rows)
    }

    /** Возвращает копию матрицы в строковом формате: массив из [rows] строк длиной [cols]. */
    fun toRows(): Array<DoubleArray> = Array(rows) { i -> DoubleArray(cols) { j -> data[i + j * rows] } }

    override fun equals(other: Any?): Boolean =
        other is DenseMatrix && rows == other.rows && cols == other.cols && data.contentEquals(other.data)

    override fun hashCode(): Int = 31 * (31 * rows + cols) + data.contentHashCode()

    /** Размеры и не более чем 6 x 6 верхних левых элементов; усечение обозначается многоточием. */
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

    companion object {
        private const val PREVIEW = 6

        /** Создаёт нулевую матрицу rows x cols; бросает [IllegalArgumentException] при отрицательных размерах. */
        fun zeros(rows: Int, cols: Int): DenseMatrix {
            require(rows >= 0 && cols >= 0) { "Размеры матрицы не могут быть отрицательными: $rows x $cols" }
            return DenseMatrix(rows, cols, DoubleArray(rows * cols))
        }

        /** Создаёт единичную матрицу n x n. */
        fun identity(n: Int): DenseMatrix {
            val m = zeros(n, n)
            for (i in 0 until n) m.data[i + i * n] = 1.0
            return m
        }

        /**
         * Строит матрицу из массива строк, копируя данные. Пустой массив даёт матрицу 0 x 0.
         * Бросает [IllegalArgumentException], если строки имеют разную длину.
         */
        fun fromRows(rows: Array<DoubleArray>): DenseMatrix {
            if (rows.isEmpty()) return DenseMatrix(0, 0, DoubleArray(0))
            val n = rows.size
            val cols = rows[0].size
            for (i in rows.indices) {
                require(rows[i].size == cols) { "Строка $i имеет длину ${rows[i].size}, ожидалось $cols" }
            }
            val data = DoubleArray(n * cols)
            for (i in 0 until n) {
                val r = rows[i]
                for (j in 0 until cols) data[i + j * n] = r[j]
            }
            return DenseMatrix(n, cols, data)
        }

        /**
         * Оборачивает готовый массив в столбцовом порядке без копирования: матрица владеет
         * переданным массивом, и все последующие изменения массива видны через неё (и наоборот).
         * Бросает [IllegalArgumentException], если размеры отрицательны или длина массива не равна `rows * cols`.
         */
        fun fromColumnMajor(rows: Int, cols: Int, data: DoubleArray): DenseMatrix = DenseMatrix(rows, cols, data)

        /** Строит матрицу rows x cols, вычисляя каждый элемент (i, j) функцией [cell]. */
        fun build(rows: Int, cols: Int, cell: (Int, Int) -> Double): DenseMatrix {
            val m = zeros(rows, cols)
            val data = m.data
            for (j in 0 until cols) {
                val base = j * rows
                for (i in 0 until rows) data[base + i] = cell(i, j)
            }
            return m
        }

        /** Создаёт диагональную матрицу n x n с элементами [d] на диагонали, где n — длина [d]. */
        fun diagonal(d: DoubleArray): DenseMatrix {
            val n = d.size
            val m = zeros(n, n)
            for (i in 0 until n) m.data[i + i * n] = d[i]
            return m
        }
    }
}
