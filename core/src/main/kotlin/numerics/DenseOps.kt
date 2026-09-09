package numerics

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Скалярные и служебные операции плотной линейной алгебры над строковыми массивами:
 * конструкторы матриц, нормы векторов и мера несимметричности. Не обращаются к
 * реализации линейной алгебры.
 *
 * Внутренние операции без проверок аргументов; проверки выполняет вызывающий код —
 * [LinearAlgebra], через который операции доступны снаружи.
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

    /** Мера несимметричности max|A - A^T| без проверок входа (их выполняет [LinearAlgebra.maxAsymmetry]). */
    fun maxAsymmetry(a: Array<DoubleArray>): Double {
        var m = 0.0
        for (i in a.indices) for (j in a.indices) m = maxOf(m, abs(a[i][j] - a[j][i]))
        return m
    }
}
