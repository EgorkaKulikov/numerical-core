package numerics

import kotlin.math.abs

// ============================================================================
// 2. КВАДРАТУРА ГАУССА--ЛЕЖАНДРА
// ============================================================================

/**
 * Составная квадратура Гаусса--Лежандра по подынтервалам. Базовый инструмент для
 * интегралов \mathcal K omega_i, M, M^{(2)}, правых частей и L2-величин.
 * Порядок выбирается выше порядка аппроксимации проектора (чтобы погрешность
 * квадратур не маскировала различие Слоан/Кулкарни; risk R3 numerical-scheme.md).
 */
class GaussLegendre(val nodesPerSub: Int = 8) {

    private val refNodes: DoubleArray
    private val refWeights: DoubleArray

    init {
        require(nodesPerSub >= 1)
        val (nodes, weights) = gaussLegendreReference(nodesPerSub)
        refNodes = nodes
        refWeights = weights
    }

    /** Интеграл f по составному разбиению breakpoints (возрастающие, с концами). */
    fun integrate(breakpoints: DoubleArray, f: (Double) -> Double): Double {
        var sum = 0.0
        for (k in 0 until breakpoints.size - 1) {
            val lo = breakpoints[k]
            val hi = breakpoints[k + 1]
            if (hi <= lo) continue
            val half = 0.5 * (hi - lo)
            val mid = 0.5 * (hi + lo)
            for (q in refNodes.indices) {
                val t = mid + half * refNodes[q]
                sum += half * refWeights[q] * f(t)
            }
        }
        return sum
    }

    /** Интеграл по одному отрезку [lo, hi]. */
    fun integrateInterval(lo: Double, hi: Double, f: (Double) -> Double): Double =
        integrate(doubleArrayOf(lo, hi), f)

    /** Эталонные узлы и веса на [-1,1] (для ручного обхода подынтервалов). */
    fun refNodesWeights(): Pair<DoubleArray, DoubleArray> = refNodes to refWeights

    companion object {
        /**
         * Узлы и веса Гаусса--Лежандра на [-1,1] для m точек (метод Ньютона по
         * нулям многочлена Лежандра P_m). Точна для многочленов степени <= 2m-1.
         */
        fun gaussLegendreReference(m: Int): Pair<DoubleArray, DoubleArray> {
            val nodes = DoubleArray(m)
            val weights = DoubleArray(m)
            for (i in 0 until (m + 1) / 2) {
                var x = Math.cos(Math.PI * (i + 0.75) / (m + 0.5))
                var dp: Double
                for (iter in 0 until 100) {
                    var p0 = 1.0
                    var p1 = x
                    for (k in 2..m) {
                        val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
                        p0 = p1; p1 = p2
                    }
                    dp = m * (x * p1 - p0) / (x * x - 1.0)
                    val dx = p1 / dp
                    x -= dx
                    if (abs(dx) < 1e-15) break
                }
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
