package numerics

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Дешёвые скалярные и служебные операции плотной линейной алгебры над строковыми
 * массивами: конструкторы матриц, нормы векторов, разложение Холецкого и мера
 * несимметричности. Не требуют бэкенда и не зависят от него.
 *
 * `internal`: деталь реализации фасада [LinearAlgebra], а не публичный API; наружу
 * операции доступны через фасад. Проверок входа здесь нет — тела содержат только
 * вычислительное ядро, а диагностические `require` выполняет фасад.
 */
internal object DenseOps {

    /** Создаёт нулевую матрицу размера rows x cols. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) }

    /** Единичная матрица размера n x n. */
    fun identity(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    /** Евклидова норма вектора. */
    fun norm2(x: DoubleArray): Double = sqrt(x.fold(0.0) { acc, v -> acc + v * v })

    /** Бесконечная (равномерная) норма вектора. */
    fun normInf(x: DoubleArray): Double = x.fold(0.0) { acc, v -> maxOf(acc, abs(v)) }

    /**
     * Разложение Холецкого A = L L^T для симметричной положительно определённой A.
     * @return нижнетреугольная L или null, если A не положительно определена.
     */
    fun cholesky(a: Array<DoubleArray>): Array<DoubleArray>? {
        val n = a.size
        val l = zeros(n, n)
        for (i in 0 until n) {
            for (j in 0..i) {
                var s = a[i][j]
                for (k in 0 until j) s -= l[i][k] * l[j][k]
                if (i == j) {
                    if (s <= 0.0) return null
                    l[i][j] = sqrt(s)
                } else {
                    l[i][j] = s / l[j][j]
                }
            }
        }
        return l
    }

    /** Мера несимметричности max|A - A^T| без проверок входа (их выполняет [LinearAlgebra.maxAsymmetry]). */
    fun maxAsymmetry(a: Array<DoubleArray>): Double {
        var m = 0.0
        for (i in a.indices) for (j in a.indices) m = maxOf(m, abs(a[i][j] - a[j][i]))
        return m
    }
}
