package numerics.backend

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
 * Бэкенд линейной алгебры на multik с нативным OpenBLAS (multik-default).
 *
 * Это бэкенд по умолчанию: тяжёлые операции делегируются нативному BLAS/LAPACK.
 * Реализация перенесена из исходного [numerics.LinearAlgebra] без изменений
 * (включая материализованное транспонирование и ручную проверку
 * вырожденности с обёрткой NaN/Inf в [solve]).
 */
object MultikCpuBackend : LinAlgBackend {

    override val name: String = "multik-cpu (OpenBLAS)"

    /**
     * Доступность определяется реальным пробным вызовом нативного решателя на
     * матрице 2x2: если multik/нативная библиотека отсутствует или не
     * загружается, возвращается false и [Backends] выполнит откат на CPU-fallback.
     */
    override fun isAvailable(): Boolean = try {
        val probe = mk.linalg.solve(
            toD2(arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 2.0))),
            toD1(doubleArrayOf(2.0, 4.0))
        )
        val r = fromD1(probe)
        r.size == 2 && r.none { it.isNaN() || it.isInfinite() }
    } catch (_: Throwable) {
        false
    }

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

    // --- Тяжёлые операции через multik/OpenBLAS ------------------------------

    override fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray =
        fromD1(toD2(a).dot(toD1(x)))

    override fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray =
        fromD1(toD2(transpose(a)).dot(toD1(y)))

    override fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        fromD2(toD2(a).dot(toD2(b)))

    override fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> {
        // Масштабируем строки A на w (WA[k][j] = w[k]*A[k][j]) и берём A^T * (WA).
        val wa = Array(a.size) { k ->
            val row = a[k]
            val wk = w[k]
            DoubleArray(row.size) { j -> wk * row[j] }
        }
        return fromD2(toD2(transpose(a)).dot(toD2(wa)))
    }

    override fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> =
        fromD2(toD2(a) + (toD2(b) * s))

    /**
     * Решение плотной СЛАУ A x = b через multik/OpenBLAS (LAPACK).
     *
     * Входные A и b не изменяются: multik копирует данные в собственные
     * ndarray. Семантика вырожденности сохранена вручную: LAPACK для точно
     * вырожденной матрицы может не бросить исключение, а вернуть NaN/Inf,
     * поэтому исключения оборачиваются, а результат дополнительно проверяется.
     * @throws IllegalStateException при вырожденности.
     */
    override fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
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
}
