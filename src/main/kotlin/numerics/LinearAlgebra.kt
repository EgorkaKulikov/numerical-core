package numerics

import kotlin.math.abs
import kotlin.math.sqrt
import numerics.backend.Backends

/**
 * Линейная алгебра над [Array]<[DoubleArray]> — тонкий фасад над подключаемым
 * бэкендом (SPI).
 *
 * Публичный API стабилен (массивы Kotlin) и не зависит от выбранной реализации.
 * Тяжёлые операции (умножения и решение СЛАУ) делегируются активному бэкенду
 * [numerics.backend.Backends.active]; по умолчанию это нативный
 * multik/OpenBLAS-бэкенд, с автоматическим откатом на чистый JVM при отсутствии
 * нативной библиотеки. Дешёвые скалярные/служебные операции (нормы, проверка
 * симметрии, разложение Холецкого как health-check, конструкторы) реализованы
 * прямо здесь. Эталоном корректности служит [ReferenceLinearAlgebra].
 *
 * Чтобы подключить новый бэкенд (например, GPU), реализуйте
 * [numerics.backend.LinAlgBackend] и зарегистрируйте его в
 * [numerics.backend.Backends] — этот фасад менять не нужно.
 */
object LinearAlgebra {

    // --- Тривиальные конструкторы (без бэкенда) ------------------------------

    /** Создаёт нулевую матрицу размера rows x cols. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = Array(rows) { DoubleArray(cols) }

    /** Единичная матрица размера n x n. */
    fun identity(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    // --- Тяжёлые операции: делегирование активному бэкенду --------------------

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matVec: пустая матрица A" }
        require(a[0].size == x.size) { "matVec: несогласованные размеры A(${a.size}x${a[0].size}) и x(${x.size})" }
        return Backends.active.matVec(a, x)
    }

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray =
        Backends.active.matTransVec(a, y)

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matMat: пустая матрица A" }
        require(b.isNotEmpty() && b[0].isNotEmpty()) { "matMat: пустая матрица B" }
        require(a[0].size == b.size) { "matMat: несогласованные размеры A(${a.size}x${a[0].size}) и B(${b.size}x${b[0].size})" }
        return Backends.active.matMat(a, b)
    }

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "atWa: пустая матрица A" }
        require(a.size == w.size) { "atWa: несогласованные размеры A(${a.size} строк) и w(${w.size})" }
        return Backends.active.atWa(a, w)
    }

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> =
        Backends.active.addScaled(a, b, s)

    /**
     * Решение плотной СЛАУ A x = b через активный бэкенд.
     *
     * Входные A и b не изменяются. Семантика вырожденности сохранена:
     * @throws IllegalStateException при вырожденности.
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "solve: пустая матрица A" }
        require(a.size == b.size) { "solve: несогласованные размеры A(${a.size} строк) и b(${b.size})" }
        return Backends.active.solve(a, b)
    }

    // --- Дешёвые скалярные/служебные операции (без бэкенда) ------------------

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

    /** Симметрия: max|A - A^T|. */
    fun maxAsymmetry(a: Array<DoubleArray>): Double {
        var m = 0.0
        for (i in a.indices) for (j in a.indices) m = maxOf(m, abs(a[i][j] - a[j][i]))
        return m
    }
}
