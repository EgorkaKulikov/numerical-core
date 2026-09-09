package numerics

import kotlin.math.abs

/**
 * Составная квадратура Гаусса–Лежандра: на каждом подынтервале разбиения применяется
 * правило с `nodesPerSub` узлами, точное для многочленов степени `2·nodesPerSub − 1`.
 *
 * Эталонные узлы и веса на `[-1, 1]` вычисляются один раз в конструкторе
 * (см. [gaussLegendreReference]) и переносятся на каждый подынтервал линейной заменой.
 *
 * @param nodesPerSub число узлов на каждом подынтервале (не меньше 1)
 */
public class GaussLegendre(public val nodesPerSub: Int = 8) {

    private val refNodes: DoubleArray
    private val refWeights: DoubleArray

    init {
        require(nodesPerSub >= 1) { "число узлов на подынтервал должно быть не меньше 1, получено $nodesPerSub" }
        val (nodes, weights) = gaussLegendreReference(nodesPerSub)
        refNodes = nodes
        refWeights = weights
    }

    /**
     * Интеграл `f` по составному разбиению `breakpoints`: точки строго возрастают,
     * первая и последняя — концы отрезка интегрирования.
     *
     * @throws IllegalArgumentException если точек меньше двух или они не строго возрастают
     */
    public fun integrate(breakpoints: DoubleArray, f: (Double) -> Double): Double {
        require(breakpoints.size >= 2) { "разбиение должно содержать не менее двух точек, получено ${breakpoints.size}" }
        for (k in 0 until breakpoints.size - 1) {
            require(breakpoints[k] < breakpoints[k + 1]) {
                "точки разбиения должны строго возрастать; нарушение между позициями $k и ${k + 1}"
            }
        }
        var sum = 0.0
        for (k in 0 until breakpoints.size - 1) {
            val lo = breakpoints[k]
            val hi = breakpoints[k + 1]
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in refNodes.indices) {
                val t = mid + half * refNodes[q]
                sum += half * refWeights[q] * f(t)
            }
        }
        return sum
    }

    /**
     * Интеграл по одному отрезку от `lo` до `hi`. Допускается `lo > hi` — тогда концы
     * переставляются и результат берётся со знаком минус; при `lo == hi` возвращается 0.
     */
    public fun integrateInterval(lo: Double, hi: Double, f: (Double) -> Double): Double =
        when {
            lo == hi -> 0.0
            lo > hi -> -integrateInterval(hi, lo, f)
            else -> integrate(doubleArrayOf(lo, hi), f)
        }

    /** Эталонные узлы и веса на `[-1, 1]` (возвращаются копии внутренних массивов). */
    public fun refNodesWeights(): Pair<DoubleArray, DoubleArray> = refNodes.copyOf() to refWeights.copyOf()

    /** Вычисление эталонных узлов и весов Гаусса–Лежандра. */
    /** Вычисление эталонных узлов и весов Гаусса–Лежандра. */
    public companion object {
        /**
         * Узлы и веса Гаусса–Лежандра на `[-1, 1]` для `m` точек: метод Ньютона по нулям
         * многочлена Лежандра `P_m` от начального приближения `cos(π(i + 3/4)/(m + 1/2))`,
         * веса `2 / ((1 − x²)·P_m'(x)²)`. Правило точно для многочленов степени `2m − 1`.
         *
         * @throws IllegalArgumentException если `m < 1`
         * @throws IllegalStateException если итерации Ньютона для какого-либо узла не сошлись
         */
        public fun gaussLegendreReference(m: Int): Pair<DoubleArray, DoubleArray> {
            require(m >= 1) { "число узлов должно быть не меньше 1, получено $m" }
            val maxIter = 100
            val nodes = DoubleArray(m)
            val weights = DoubleArray(m)
            for (i in 0 until (m + 1) / 2) {
                var x = Math.cos(Math.PI * (i + 0.75) / (m + 0.5))
                var dp: Double
                var converged = false
                for (iter in 0 until maxIter) {
                    var p0 = 1.0
                    var p1 = x
                    for (k in 2..m) {
                        val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                        p0 = p1; p1 = p2
                    }
                    dp = m * (x * p1 - p0) / (x * x - 1.0)
                    val dx = p1 / dp
                    x -= dx
                    if (abs(dx) < 1e-15) {
                        converged = true
                        break
                    }
                }
                check(converged) { "итерации Ньютона для узла $i не сошлись за $maxIter шагов" }
                var p0 = 1.0
                var p1 = x
                for (k in 2..m) {
                    val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                    p0 = p1; p1 = p2
                }
                dp = m * (x * p1 - p0) / (x * x - 1.0)
                val w = 2.0 / ((1.0 - x * x) * dp * dp)
                nodes[i] = -x
                nodes[m - 1 - i] = x
                weights[i] = w
                weights[m - 1 - i] = w
            }
            return nodes to weights
        }
    }
}
