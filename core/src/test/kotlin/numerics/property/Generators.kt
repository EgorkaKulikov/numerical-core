package numerics.property

import net.jqwik.api.Arbitraries
import net.jqwik.api.Arbitrary
import numerics.Conditioning
import numerics.DenseMatrix
import numerics.LinearAlgebra
import numerics.backend.Backends

/**
 * Генераторы jqwik для property-based тестов: квадратные матрицы общего вида,
 * симметричные, SPD, векторы и масштабирование для проверки относительности порогов.
 */
object Generators {

    /** Граница числа обусловленности, выше которой матрица считается «почти вырожденной» и отсеивается. */
    const val MAX_COND = 1e12

    private fun entries(count: Int): Arbitrary<DoubleArray> =
        Arbitraries.doubles().between(-10.0, 10.0).array(DoubleArray::class.java).ofSize(count)

    /** Квадратная матрица n×n со случайными элементами из [-10, 10] (без сдвига диагонали). */
    fun rawSquare(minN: Int = 1, maxN: Int = 24): Arbitrary<DenseMatrix> =
        Arbitraries.integers().between(minN, maxN).flatMap { n ->
            entries(n * n).map { DenseMatrix.fromColumnMajor(n, n, it) }
        }

    /**
     * Квадратная матрица общего вида: половина матриц получает сдвиг диагонали n·10 (диагональное
     * преобладание), половина — нет. Почти вырожденные (cond∞ ≥ [MAX_COND]) отсеиваются.
     */
    fun squareGeneral(minN: Int = 1, maxN: Int = 24): Arbitrary<DenseMatrix> =
        rawSquare(minN, maxN).flatMap { a ->
            Arbitraries.of(true, false).map { dominant ->
                if (dominant) {
                    val b = a.copy()
                    for (i in 0 until b.rows) b[i, i] = b[i, i] + b.rows * 10.0
                    b
                } else {
                    a
                }
            }
        }.filter { a ->
            val c = Conditioning.conditionEstimate(a, backend = Backends.java()).condInf
            c.isFinite() && c < MAX_COND
        }

    /** Симметричная положительно определённая матрица Bᵀ·B + n·I. */
    fun spd(minN: Int = 1, maxN: Int = 24): Arbitrary<DenseMatrix> =
        rawSquare(minN, maxN).map { b ->
            val n = b.rows
            val ones = DoubleArray(n) { 1.0 }
            val btb = LinearAlgebra.atWa(b, ones, Backends.java())
            val s = DenseMatrix.build(n, n) { i, j -> 0.5 * (btb[i, j] + btb[j, i]) + if (i == j) n.toDouble() else 0.0 }
            s
        }

    /** Симметричная матрица (B + Bᵀ)/2. */
    fun symmetric(minN: Int = 1, maxN: Int = 24): Arbitrary<DenseMatrix> =
        rawSquare(minN, maxN).map { b -> DenseMatrix.build(b.rows, b.rows) { i, j -> 0.5 * (b[i, j] + b[j, i]) } }

    /** Вектор длины n с элементами из [-10, 10]. */
    fun vector(n: Int): Arbitrary<DoubleArray> = entries(n)

    /** Матрица и согласованный по размеру вектор правой части. */
    fun systemGeneral(minN: Int = 1, maxN: Int = 24): Arbitrary<Pair<DenseMatrix, DoubleArray>> =
        squareGeneral(minN, maxN).flatMap { a -> vector(a.rows).map { a to it } }

    /** Масштабы для проверки относительности порогов. */
    val SCALES: List<Double> = listOf(1e-8, 1.0, 1e8)

    /** Все элементы матрицы умножены на один из масштабов [scales]. */
    fun scaled(arb: Arbitrary<DenseMatrix>, scales: List<Double> = SCALES): Arbitrary<Pair<DenseMatrix, Double>> =
        arb.flatMap { a -> Arbitraries.of(scales).map { c -> scale(a, c) to c } }

    fun scale(a: DenseMatrix, c: Double): DenseMatrix =
        DenseMatrix.fromColumnMajor(a.rows, a.cols, DoubleArray(a.data.size) { a.data[it] * c })

    fun scale(x: DoubleArray, c: Double): DoubleArray = DoubleArray(x.size) { x[it] * c }
}
