package numerics

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Эталонная (reference) реализация линейной алгебры на чистом Kotlin.
 *
 * Это «оракул корректности»: точная копия исходной ручной реализации
 * [LinearAlgebra], сохранённая для перекрёстной проверки оптимизированного
 * multik/OpenBLAS-бэкенда. Не используется в боевых вычислениях, только в тестах.
 */
object ReferenceLinearAlgebra {

    /** Создаёт нулевую матрицу размера rows x cols. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) }

    /** Единичная матрица размера n x n. */
    fun identity(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray {
        val m = a.size
        val out = DoubleArray(m)
        for (i in 0 until m) {
            var s = 0.0
            val row = a[i]
            for (j in x.indices) s += row[j] * x[j]
            out[i] = s
        }
        return out
    }

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray {
        val m = a.size
        val n = a[0].size
        val out = DoubleArray(n)
        for (i in 0 until m) {
            val row = a[i]
            val yi = y[i]
            for (j in 0 until n) out[j] += row[j] * yi
        }
        return out
    }

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val m = a.size
        val k = b.size
        val p = b[0].size
        val out = zeros(m, p)
        for (i in 0 until m) {
            val ai = a[i]
            val oi = out[i]
            for (l in 0 until k) {
                val ail = ai[l]
                if (ail == 0.0) continue
                val bl = b[l]
                for (j in 0 until p) oi[j] += ail * bl[j]
            }
        }
        return out
    }

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> {
        val m = a.size
        val n = a[0].size
        val out = zeros(n, n)
        for (k in 0 until m) {
            val row = a[k]
            val wk = w[k]
            for (i in 0 until n) {
                val rwi = wk * row[i]
                if (rwi == 0.0) continue
                val outI = out[i]
                for (j in 0 until n) outI[j] += rwi * row[j]
            }
        }
        return out
    }

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> {
        val out = Array(a.size) { a[it].copyOf() }
        for (i in a.indices) for (j in a[i].indices) out[i][j] += s * b[i][j]
        return out
    }

    /** Евклидова норма вектора. */
    fun norm2(x: DoubleArray): Double = sqrt(x.fold(0.0) { acc, v -> acc + v * v })

    /** Бесконечная (равномерная) норма вектора. */
    fun normInf(x: DoubleArray): Double = x.fold(0.0) { acc, v -> maxOf(acc, abs(v)) }

    /**
     * Решение плотной СЛАУ A x = b методом LU с частичным выбором ведущего
     * элемента. Матрица A и вектор b не изменяются (работаем на копиях).
     * @throws IllegalStateException при вырожденности.
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val n = a.size
        val lu = Array(n) { a[it].copyOf() }
        val x = b.copyOf()
        val piv = IntArray(n) { it }
        for (col in 0 until n) {
            var pivRow = col
            var pivVal = abs(lu[col][col])
            for (r in col + 1 until n) {
                val v = abs(lu[r][col])
                if (v > pivVal) { pivVal = v; pivRow = r }
            }
            if (pivVal < 1e-300) error("LU: матрица вырождена (col=$col)")
            if (pivRow != col) {
                val t = lu[col]; lu[col] = lu[pivRow]; lu[pivRow] = t
                val tx = x[col]; x[col] = x[pivRow]; x[pivRow] = tx
                val tp = piv[col]; piv[col] = piv[pivRow]; piv[pivRow] = tp
            }
            val pivotR = lu[col]
            val pivot = pivotR[col]
            for (r in col + 1 until n) {
                val factor = lu[r][col] / pivot
                lu[r][col] = factor
                val rowR = lu[r]
                for (c in col + 1 until n) rowR[c] -= factor * pivotR[c]
                x[r] -= factor * x[col]
            }
        }
        // обратный ход
        for (i in n - 1 downTo 0) {
            var s = x[i]
            val row = lu[i]
            for (j in i + 1 until n) s -= row[j] * x[j]
            x[i] = s / row[i]
        }
        return x
    }

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

    /** Симметрия: max|A - A^T|. */
    fun maxAsymmetry(a: Array<DoubleArray>): Double {
        var m = 0.0
        for (i in a.indices) for (j in a.indices) m = maxOf(m, abs(a[i][j] - a[j][i]))
        return m
    }
}

