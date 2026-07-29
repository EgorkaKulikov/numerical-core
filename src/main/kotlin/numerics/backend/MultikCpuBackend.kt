package numerics.backend

import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.solve
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get

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

    /**
     * Array<DoubleArray> (rows x cols) -> D2 NDArray.
     *
     * Собираем плоский C-order `DoubleArray` и создаём ndarray на примитивном
     * пути `mk.ndarray(flat, rows, cols)` — без боксинга элементов в
     * java.lang.Double и без промежуточных List<Double>.
     */
    private fun toD2(a: Array<DoubleArray>): D2Array<Double> {
        val rows = a.size
        val cols = if (rows == 0) 0 else a[0].size
        val flat = DoubleArray(rows * cols)
        var offset = 0
        for (row in a) {
            System.arraycopy(row, 0, flat, offset, cols)
            offset += cols
        }
        return mk.ndarray(flat, rows, cols)
    }

    /**
     * A^T как contiguous D2 NDArray (n x m для A размера m x n).
     *
     * Материализуем транспонирование в плоский C-order `DoubleArray` без
     * боксинга и строим ndarray через `mk.ndarray(flatT, cols, rows)`.
     * Причина: в multik 0.2.3 `dot` некорректно учитывает strides
     * транспонированной VIEW для НЕквадратных матриц (даёт неверный результат),
     * поэтому передаём в BLAS уже contiguous-массив.
     */
    private fun toD2Transposed(a: Array<DoubleArray>): D2Array<Double> {
        val rows = a.size
        val cols = if (rows == 0) 0 else a[0].size
        val flatT = DoubleArray(rows * cols)
        for (i in 0 until rows) {
            val row = a[i]
            var idx = i
            for (j in 0 until cols) {
                flatT[idx] = row[j]
                idx += rows
            }
        }
        return mk.ndarray(flatT, cols, rows)
    }

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

    // --- Тяжёлые операции через multik/OpenBLAS ------------------------------

    override fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray =
        fromD1(toD2(a).dot(toD1(x)))

    override fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray =
        // A^T материализуем в contiguous ndarray (без боксинга): view-транспонирование
        // ломает dot для неквадратных матриц в multik 0.2.3.
        fromD1(toD2Transposed(a).dot(toD1(y)))

    override fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        fromD2(toD2(a).dot(toD2(b)))

    override fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> {
        // Масштабируем строки A на w (WA[k][j] = w[k]*A[k][j]) и берём A^T * (WA).
        val wa = Array(a.size) { k ->
            val row = a[k]
            val wk = w[k]
            DoubleArray(row.size) { j -> wk * row[j] }
        }
        // A^T материализуем в contiguous ndarray (без боксинга) — см. toD2Transposed.
        return fromD2(toD2Transposed(a).dot(toD2(wa)))
    }

    /**
     * Поэлементная сумма `A + s*B` — намеренно СОБСТВЕННАЯ реализация на чистом JVM,
     * без обращения к multik.
     *
     * ОБОСНОВАНИЕ ИЗМЕРЕНИЯМИ (бенчмарк `runBenchmark`, исследование (C), три прогона
     * на Apple Silicon; колонка multik/reference — во сколько раз нативный путь МЕДЛЕННЕЕ):
     *
     * | размер | прогон 1 | прогон 2 | прогон 3 |
     * |--------|----------|----------|----------|
     * | 16     | 4.25     | 4.24     | 4.20     |
     * | 64     | 3.93     | 3.78     | 3.07     |
     * | 256    | 13.67    | 1.00     | 1.49     |
     * | 1024   | 3.56     | 2.67     | 1.91     |
     *
     * Нативный путь проигрывает на ВСЕХ размерностях без исключения, поэтому порог
     * переключения не нужен — выгоднее не обращаться к multik вовсе.
     *
     * Причина: операция поэлементная и ограничена пропускной способностью памяти, для
     * неё нет ядра BLAS, которое окупило бы накладные расходы. Прежняя реализация
     * `fromD2(toD2(a) + (toD2(b) * s))` создавала ЧЕТЫРЕ временных NDArray (два на
     * конвертацию входов, по одному на умножение и сложение) и дополнительно читала
     * результат поэлементно в [fromD2] — ради работы, стоящей одного прохода по массиву.
     *
     * Результат БИТОВО совпадает с прежним: порядок арифметики тот же (`a + s*b`),
     * а умножение IEEE-754 коммутативно, поэтому `s*b` и `b*s` дают идентичные биты.
     * Это проверено характеризационным тестом `EhCharacterizationTest`, прошедшим без
     * пересъёмки эталона.
     */
    override fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> {
        val rows = a.size
        val cols = if (rows == 0) 0 else a[0].size
        return Array(rows) { i ->
            val rowA = a[i]
            val rowB = b[i]
            DoubleArray(cols) { j -> rowA[j] + s * rowB[j] }
        }
    }

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
