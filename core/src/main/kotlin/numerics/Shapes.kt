package numerics

import kotlin.math.abs

/**
 * Единая проверка форм входных данных для [LinearAlgebra] и [Conditioning].
 * Каждая проверка завершается [IllegalArgumentException] с русским сообщением,
 * в котором названы фактические размеры.
 */
internal object Shapes {

    /** Матрица должна иметь хотя бы одну строку и один столбец. */
    fun requireNonEmpty(a: DenseMatrix, what: String = "матрица") {
        require(a.rows > 0 && a.cols > 0) { "$what не должна быть пустой, получено ${a.rows}×${a.cols}" }
    }

    /** Матрица должна быть непустой и квадратной. */
    fun requireSquare(a: DenseMatrix, what: String = "матрица") {
        requireNonEmpty(a, what)
        require(a.isSquare) { "$what должна быть квадратной, получено ${a.rows}×${a.cols}" }
    }

    /** Длина вектора должна равняться [n]. */
    fun requireVectorLength(v: DoubleArray, n: Int, what: String) {
        require(v.size == n) { "длина вектора $what должна быть $n, получено ${v.size}" }
    }

    /** Обе матрицы непусты, число столбцов левой равно числу строк правой. */
    fun requireMultiplicable(a: DenseMatrix, b: DenseMatrix) {
        requireNonEmpty(a, "левая матрица")
        requireNonEmpty(b, "правая матрица")
        require(a.cols == b.rows) {
            "размеры не согласованы для умножения: ${a.rows}×${a.cols} и ${b.rows}×${b.cols}"
        }
    }

    /** Размеры двух матриц совпадают. */
    fun requireSameShape(a: DenseMatrix, b: DenseMatrix) {
        require(a.rows == b.rows && a.cols == b.cols) {
            "размеры матриц должны совпадать, получено ${a.rows}×${a.cols} и ${b.rows}×${b.cols}"
        }
    }

    /** Все элементы матрицы — конечные числа. */
    fun requireFinite(a: DenseMatrix, what: String) {
        require(a.data.all { it.isFinite() }) { "$what содержит нечисловые значения (NaN или бесконечность)" }
    }

    /** Все компоненты вектора — конечные числа. */
    fun requireFinite(v: DoubleArray, what: String) {
        require(v.all { it.isFinite() }) { "$what содержит нечисловые значения (NaN или бесконечность)" }
    }

    /**
     * Квадратная матрица симметрична с точностью до округления:
     * `max|A − Aᵀ| ≤ 1e-12 · ‖A‖∞`; норму [normInf] передаёт вызывающий.
     */
    fun requireSymmetric(a: DenseMatrix, normInf: Double, what: String = "матрица") {
        requireSquare(a, what)
        val n = a.rows
        val d = a.data
        var asym = 0.0
        for (j in 0 until n) for (i in j + 1 until n) {
            asym = maxOf(asym, abs(d[i + j * n] - d[j + i * n]))
        }
        require(asym <= 1e-12 * maxOf(normInf, Double.MIN_VALUE)) {
            "$what должна быть симметричной: max|A−Aᵀ| = $asym"
        }
    }
}
