package numerics

/**
 * Convergence orders from an error table: `p_h = log2(E_h / E_{h/2})` over adjacent rows.
 *
 * The functions operate on ready-made error values and know nothing about how or on which grid
 * they were measured. The variants with a reliability threshold are [reliableOrders] and [reliableConstCh].
 *
 * Returns [Double.NaN] for the last row (there is no next one) and for any pair in which either
 * error is non-finite or non-positive: for `E = 0` or `E = ∞` the ratio gives 0 or infinity,
 * and the order is undefined at machine precision.
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
 * Constant `C_h = E_h / h^p` for a known order `p`.
 *
 * @throws IllegalArgumentException if the step `h` is not positive
 */
public fun constCh(eh: Double, h: Double, p: Double): Double {
    require(h > 0) { "grid step must be positive, got $h" }
    return eh / Math.pow(h, p)
}
