package numerics

/**
 * Порядки сходимости по таблице погрешностей: `p_h = log2(E_h / E_{h/2})` по соседним строкам.
 *
 * Функции работают с готовыми числами погрешностей и не знают, чем и на какой сетке они
 * измерены. Вариант с порогом достоверности — [reliableOrders] и [reliableConstCh].
 *
 * Возвращает [Double.NaN] для последней строки (следующей нет) и для пары, в которой любая из
 * погрешностей не конечна или не положительна: при `E = 0` или `E = ∞` отношение даёт 0 или
 * бесконечность, и порядок не определён на машинной точности.
 */
public fun orders(errs: List<Double>): List<Double> =
    errs.indices.map { i ->
        when {
            i + 1 >= errs.size -> Double.NaN
            !errs[i].isFinite() || !errs[i + 1].isFinite() -> Double.NaN
            errs[i] <= 0.0 || errs[i + 1] <= 0.0 -> Double.NaN
            else -> Math.log(errs[i] / errs[i + 1]) / Math.log(2.0)
        }
    }

/**
 * Константа `C_h = E_h / h^p` при известном порядке `p`.
 *
 * @throws IllegalArgumentException если шаг `h` не положителен
 */
public fun constCh(eh: Double, h: Double, p: Double): Double {
    require(h > 0) { "шаг сетки должен быть положительным, получено $h" }
    return eh / Math.pow(h, p)
}
