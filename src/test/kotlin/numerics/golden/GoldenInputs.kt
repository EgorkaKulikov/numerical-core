package numerics.golden

import numerics.ConditionEstimate
import numerics.ForwardError
import numerics.LinearAlgebra
import numerics.Measured
import numerics.golden.GoldenIo.dbl
import java.util.Random
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sin

/**
 * Детерминированные входы для golden-эталонов. Каждый генератор создаёт **свой**
 * `Random` с seed, зависящим только от `n`, поэтому результат не зависит от порядка вызовов.
 */
object GoldenInputs {

    /** Диагонально доминирующая: `a[i][j] ∈ (−1, 1)`, `a[i][i] += n`. */
    fun dd(n: Int): Array<DoubleArray> {
        val r = Random(1000L + n)
        val a = Array(n) { DoubleArray(n) }
        for (i in 0 until n) for (j in 0 until n) a[i][j] = r.nextDouble() * 2 - 1
        for (i in 0 until n) a[i][i] += n.toDouble()
        return a
    }

    /** `dd(n)` с циклическим сдвигом строк на `n/2` — заставляет LU переставлять строки. */
    fun perm(n: Int): Array<DoubleArray> {
        val a = dd(n)
        val s = n / 2
        return Array(n) { i -> a[(i + s) % n].copyOf() }
    }

    fun hilbert(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> 1.0 / (i + j + 1) } }

    /** Симметричная: `(b + bᵀ)/2`, в `[i][j]` и `[j][i]` записано одно и то же значение. */
    fun sym(n: Int): Array<DoubleArray> {
        val r = Random(1000L + n)
        val b = Array(n) { DoubleArray(n) }
        for (i in 0 until n) for (j in 0 until n) b[i][j] = r.nextDouble() * 2 - 1
        val a = Array(n) { DoubleArray(n) }
        for (i in 0 until n) for (j in i until n) {
            val v = (b[i][j] + b[j][i]) / 2
            a[i][j] = v
            a[j][i] = v
        }
        return a
    }

    /** SPD: `Bᵀ·B + n·I` через `LinearAlgebra.atWa(b, 1)`. */
    fun spd(n: Int): Array<DoubleArray> {
        val r = Random(1000L + n)
        val b = Array(n) { DoubleArray(n) }
        for (i in 0 until n) for (j in 0 until n) b[i][j] = r.nextDouble() * 2 - 1
        val ones = DoubleArray(n) { 1.0 }
        val btb = LinearAlgebra.atWa(b, ones)
        val a = Array(n) { i -> btb[i].copyOf() }
        for (i in 0 until n) a[i][i] += n.toDouble()
        return a
    }

    fun vec(n: Int): DoubleArray {
        val r = Random(2000L + n)
        return DoubleArray(n) { r.nextDouble() * 2 - 1 }
    }

    fun posVec(n: Int): DoubleArray {
        val r = Random(3000L + n)
        return DoubleArray(n) { r.nextDouble() + 0.5 }
    }

    fun transpose(a: Array<DoubleArray>): Array<DoubleArray> {
        val rows = a.size
        val cols = if (rows == 0) 0 else a[0].size
        return Array(cols) { j -> DoubleArray(rows) { i -> a[i][j] } }
    }

    fun matrixOf(cls: String, n: Int): Array<DoubleArray> = when (cls) {
        "dd" -> dd(n)
        "perm" -> perm(n)
        "hilbert" -> hilbert(n)
        "sym" -> sym(n)
        "spd" -> spd(n)
        else -> error("Неизвестный класс матриц: $cls")
    }

    // ---- Перечни случаев (общие для генератора и проверок) ---------------------

    /** Пара (класс, n); ключ случая — `"$cls-$n"`. */
    data class MatrixCase(val cls: String, val n: Int) {
        val key: String get() = "$cls-$n"
        fun matrix(): Array<DoubleArray> = matrixOf(cls, n)
    }

    val solveCases: List<MatrixCase> =
        listOf("dd", "perm").flatMap { cls -> listOf(1, 2, 3, 5, 8, 16, 32, 64).map { MatrixCase(cls, it) } } +
            listOf(2, 3, 5, 8).map { MatrixCase("hilbert", it) }

    val matOpsCases: List<MatrixCase> = listOf(1, 2, 3, 5, 8, 16).map { MatrixCase("dd", it) }

    val choleskyCases: List<MatrixCase> = listOf(1, 2, 3, 5, 8, 16, 32).map { MatrixCase("spd", it) }

    val conditioningCases: List<MatrixCase> =
        listOf(1, 2, 3, 5, 8, 16, 32).map { MatrixCase("dd", it) } +
            listOf(2, 3, 5, 8).map { MatrixCase("hilbert", it) }

    val eigenCases: List<MatrixCase> = listOf(1, 2, 3, 5, 8, 16, 32).map { MatrixCase("sym", it) }

    val conditionSymmetricCases: List<MatrixCase> = listOf(1, 2, 3, 5, 8, 16, 32).map { MatrixCase("spd", it) }

    /** Допуск для сравнения результатов LU-семейства: матрицы Гильберта — грубее. */
    fun tolFor(case: MatrixCase): Double = if (case.cls == "hilbert") HILBERT_TOL else DEFAULT_TOL

    const val DEFAULT_TOL = 1e-12
    const val HILBERT_TOL = 1e-6

    // ---- Квадратура -------------------------------------------------------------

    val referenceOrders: List<Int> = (1..32).toList()

    val integrateOrders: List<Int> = listOf(2, 4, 8, 12)

    val breakpoints: List<Pair<String, DoubleArray>> = listOf(
        "unit" to doubleArrayOf(0.0, 1.0),
        "split4" to doubleArrayOf(0.0, 0.25, 0.5, 1.0),
        "wide" to doubleArrayOf(-1.0, 0.3, 2.7),
    )

    val integrands: List<Pair<String, (Double) -> Double>> = listOf(
        "exp" to { t: Double -> exp(t) },
        "runge" to { t: Double -> 1.0 / (1.0 + t * t) },
        "t7" to { t: Double -> t.pow(7) },
        "sin10t" to { t: Double -> sin(10.0 * t) },
    )

    class QuadCase(val m: Int, val bpName: String, val bp: DoubleArray, val fName: String, val f: (Double) -> Double) {
        val key: String get() = "m$m-$bpName-$fName"
    }

    val integrateCases: List<QuadCase> = integrateOrders.flatMap { m ->
        breakpoints.flatMap { (bpName, bp) ->
            integrands.map { (fName, f) -> QuadCase(m, bpName, bp, fName, f) }
        }
    }

    // ---- Чистые функции -------------------------------------------------------

    val feConds: List<Double> = listOf(1.0, 1e3, 1e8, 1e12, 1e16, Double.POSITIVE_INFINITY)
    val feResiduals: List<Double> = listOf(0.0, 1e-12, 1e-6)
    val feOmegas: List<Double> = listOf(0.0, 1e-16, 1e-10, 1e-3)

    data class FeCase(val cond: Double, val residual: Double, val omega: Double) {
        val key: String get() = "cond=$cond;residual=$residual;omega=$omega"
    }

    val feCases: List<FeCase> = feConds.flatMap { c -> feResiduals.flatMap { r -> feOmegas.map { o -> FeCase(c, r, o) } } }

    val measuredValues: List<Double> =
        listOf(0.0, 1e-14, 1e-13, 1e-12, 1.0, -1.0, Double.NaN, Double.POSITIVE_INFINITY)
    val measuredThresholds: List<Double> = listOf(1e-13, 1e-10)

    data class MeasuredCase(val value: Double, val threshold: Double) {
        val key: String get() = "v=$value;thr=$threshold"
    }

    val measuredCases: List<MeasuredCase> =
        measuredValues.flatMap { v -> measuredThresholds.map { t -> MeasuredCase(v, t) } }

    val orderLists: List<List<Double>> = listOf(
        listOf(1.0, 0.5, 0.25),
        listOf(1.0, 0.25, 0.0625),
        listOf(1e-3, 1e-13, 1e-14),
        listOf(1.0, 0.0, 0.5),
        listOf(1.0, -0.5),
        listOf(1.0, Double.NaN),
        listOf(Double.POSITIVE_INFINITY, 1.0),
        listOf(1.0),
        emptyList(),
    )

    fun orderListKey(list: List<Double>): String = "[" + list.joinToString(", ") + "]"

    data class ConstChCase(val eh: Double, val h: Double, val p: Double) {
        val key: String get() = "eh=$eh;h=$h;p=$p"
    }

    val constChCases: List<ConstChCase> = listOf(
        ConstChCase(1e-3, 0.1, 2.0),
        ConstChCase(1e-14, 0.1, 2.0),
        ConstChCase(0.0, 0.5, 1.0),
    )

    data class RatioCase(val a: Double, val b: Double) {
        val key: String get() = "a=$a;b=$b"
    }

    val ratioCases: List<RatioCase> = listOf(
        RatioCase(1.0, 0.25),
        RatioCase(1e-14, 1e-15),
        RatioCase(1.0, 0.0),
        RatioCase(0.0, 1.0),
    )

    // ---- Сериализация типов результатов -----------------------------------------

    fun conditionToJson(c: ConditionEstimate): Map<String, Any?> = linkedMapOf(
        "condInf" to dbl(c.condInf),
        "inversionResidual" to dbl(c.inversionResidual),
        "tolerance" to dbl(c.tolerance),
        "isReliable" to c.isReliable,
    )

    fun forwardErrorToJson(fe: ForwardError): Map<String, Any?> = when (fe) {
        is ForwardError.Bounded -> linkedMapOf(
            "type" to "Bounded",
            "backwardError" to dbl(fe.backwardError),
            "cond" to dbl(fe.cond),
            "relativeBound" to dbl(fe.relativeBound),
        )
        is ForwardError.NoFiniteBound -> linkedMapOf(
            "type" to "NoFiniteBound",
            "backwardError" to dbl(fe.backwardError),
        )
        is ForwardError.Unreliable -> linkedMapOf(
            "type" to "Unreliable",
            "backwardError" to dbl(fe.backwardError),
            "condition" to conditionToJson(fe.condition),
        )
    }

    fun measuredToJson(m: Measured): Map<String, Any?> = when (m) {
        is Measured.Reliable -> linkedMapOf("type" to "Reliable", "value" to dbl(m.value))
        is Measured.AtNoiseLevel -> linkedMapOf(
            "type" to "AtNoiseLevel",
            "value" to dbl(m.value),
            "threshold" to dbl(m.threshold),
        )
    }
}
