package numerics

import kotlin.math.abs
import kotlin.math.sqrt
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.solve
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times

/**
 * Линейная алгебра над [Array]<[DoubleArray]>.
 *
 * Публичный API стабилен (массивы Kotlin), однако тяжёлые операции
 * (умножения и решение СЛАУ) делегируются библиотеке multik с нативным
 * бэкендом OpenBLAS (multik-default). Скалярные операции (нормы, проверка
 * симметрии, разложение Холецкого как health-check) оставлены на чистом
 * Kotlin. Эталоном корректности служит [ReferenceLinearAlgebra].
 */
object LinearAlgebra {

    // --- Конвертеры массивов Kotlin <-> NDArray multik -----------------------

    /** Array<DoubleArray> (rows x cols) -> D2 NDArray. */
    private fun toD2(a: Array<DoubleArray>): D2Array<Double> =
        mk.ndarray(a.map { it.toList() })

    /** D1 NDArray -> DoubleArray. */
    private fun toD1(x: DoubleArray): D1Array<Double> = mk.ndarray(x)

    /** D2 NDArray -> Array<DoubleArray>. */
    private fun fromD2(m: D2Array<Double>): Array<DoubleArray> {
        val rows = m.shape[0]
        val cols = m.shape[1]
        return Array(rows) { i -> DoubleArray(cols) { j -> m[i, j] } }
    }

    /** D1 NDArray -> DoubleArray. */
    private fun fromD1(v: D1Array<Double>): DoubleArray {
        val n = v.shape[0]
        return DoubleArray(n) { i -> v[i] }
    }

    /** Явное (материализованное) транспонирование A (m x n) -> (n x m). */
    private fun transpose(a: Array<DoubleArray>): Array<DoubleArray> {
        val m = a.size
        val n = a[0].size
        return Array(n) { j -> DoubleArray(m) { i -> a[i][j] } }
    }

    // --- Тривиальные конструкторы (без multik) -------------------------------

    /** Создаёт нулевую матрицу размера rows x cols. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) }

    /** Единичная матрица размера n x n. */
    fun identity(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    // --- Тяжёлые операции через multik/OpenBLAS ------------------------------

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray =
        fromD1(toD2(a).dot(toD1(x)))

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray =
        fromD1(toD2(transpose(a)).dot(toD1(y)))

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        fromD2(toD2(a).dot(toD2(b)))

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> {
        // Масштабируем строки A на w (WA[k][j] = w[k]*A[k][j]) и берём A^T * (WA).
        val wa = Array(a.size) { k ->
            val row = a[k]
            val wk = w[k]
            DoubleArray(row.size) { j -> wk * row[j] }
        }
        return fromD2(toD2(transpose(a)).dot(toD2(wa)))
    }

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> =
        fromD2(toD2(a) + (toD2(b) * s))

    /** Евклидова норма вектора. */
    fun norm2(x: DoubleArray): Double = sqrt(x.fold(0.0) { acc, v -> acc + v * v })

    /** Бесконечная (равномерная) норма вектора. */
    fun normInf(x: DoubleArray): Double = x.fold(0.0) { acc, v -> maxOf(acc, abs(v)) }

    /**
     * Решение плотной СЛАУ A x = b через multik/OpenBLAS (LAPACK).
     *
     * Входные A и b не изменяются: multik копирует данные в собственные
     * ndarray. Семантика вырожденности сохранена вручную: LAPACK для точно
     * вырожденной матрицы может не бросить исключение, а вернуть NaN/Inf,
     * поэтому исключения оборачиваются, а результат дополнительно проверяется.
     * @throws IllegalStateException при вырожденности.
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val result = try {
            fromD1(mk.linalg.solve(toD2(a), toD1(b)))
        } catch (e: Exception) {
            throw IllegalStateException("LU/solve: матрица вырождена", e)
        }
        for (v in result) {
            if (v.isNaN() || v.isInfinite()) error("LU/solve: матрица вырождена")
        }
        return result
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
