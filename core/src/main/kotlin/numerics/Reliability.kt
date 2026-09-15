package numerics

import kotlin.math.abs

/**
 * Threshold below which a value is treated as rounding noise rather than a measurement.
 * Chosen two to three orders of magnitude above the double machine epsilon (2.2e-16) to separate
 * the method error from the rounding error accumulated in sums of length 10²–10³.
 *
 * A value is first classified ([measured]); the ratio ([ratio]) and the convergence order
 * ([orderOrNull], [reliableOrders]) are by construction not computed if either argument
 * lies at noise level.
 */
public const val MACHINE_NOISE_THRESHOLD: Double = 1e-13

/**
 * A value together with a verdict on its reliability.
 *
 * The type is deliberately `sealed`: going from the number to a derived quantity
 * (ratio, convergence order) is possible only through explicit pattern matching,
 * so an unreliable value cannot be used by accident.
 */
public sealed interface Measured {
    /** The numeric value itself — always available, e.g. for printing with a marker. */
    public val value: Double

    /** The value exceeds the reliability threshold and may take part in computations. */
    public data class Reliable(override val value: Double) : Measured

    /**
     * The value does not exceed the threshold [threshold] (or is not finite) — it is noise.
     * @property threshold the reliability threshold the value was compared against.
     */
    public data class AtNoiseLevel(override val value: Double, val threshold: Double) : Measured
}

/**
 * Classifies a value against a reliability threshold.
 *
 * Non-finite values ([Double.NaN], infinities) are treated as unreliable:
 * they arise from division by zero and from overflow and are not measurements.
 *
 * @throws IllegalArgumentException if `threshold <= 0` — a zero threshold would mean
 *   no check at all, i.e. exactly the mistake this type prevents.
 */
public fun measured(value: Double, threshold: Double = MACHINE_NOISE_THRESHOLD): Measured {
    require(threshold > 0.0) { "measured: threshold must be > 0, got $threshold" }
    return if (value.isFinite() && abs(value) >= threshold) {
        Measured.Reliable(value)
    } else {
        Measured.AtNoiseLevel(value, threshold)
    }
}

/**
 * Ratio of two values, or `null` if either of them is unreliable.
 *
 * `null` here means "the quantity is undefined", not "could not be computed":
 * the ratio of numbers at rounding-noise level carries no information about the method.
 */
public fun ratio(numerator: Measured, denominator: Measured): Double? =
    if (numerator is Measured.Reliable && denominator is Measured.Reliable) {
        numerator.value / denominator.value
    } else {
        null
    }

/**
 * Empirical convergence order `log2(E_h / E_{h/2})` from a pair of reliable errors,
 * or `null`.
 *
 * `null` is returned if either error is at noise level ([ratio]) or their ratio is
 * non-positive (the logarithm is undefined).
 */
public fun orderOrNull(coarse: Measured, fine: Measured): Double? {
    val r = ratio(coarse, fine) ?: return null
    return if (r <= 0.0) null else Math.log(r) / Math.log(2.0)
}

/**
 * Column of convergence orders with the reliability threshold applied —
 * the variant of [orders] that refuses to compute an order from noise.
 *
 * Relation to [orders]: both functions yield "undefined" for the last row
 * (there is simply no next row). The difference is that [orders] computes the order
 * from any positive values, whereas this function uses only values not below [threshold];
 * "undefined" is encoded here as `null` rather than [Double.NaN], so the value cannot be
 * accidentally fed into arithmetic. For printing in a single format
 * `reliableOrders(errs).map { it ?: Double.NaN }` suffices, and when all errors are above
 * the threshold the result coincides with `orders(errs)`.
 */
public fun reliableOrders(errs: List<Double>, threshold: Double = MACHINE_NOISE_THRESHOLD): List<Double?> {
    val m = errs.map { measured(it, threshold) }
    return m.indices.map { i -> if (i + 1 < m.size) orderOrNull(m[i], m[i + 1]) else null }
}

/**
 * Constant `C_h = E_h / h^p` from a reliable error, or `null` —
 * the variant of [constCh] with the reliability threshold applied.
 *
 * @throws IllegalArgumentException if the step `h` is not positive
 */
public fun reliableConstCh(eh: Measured, h: Double, p: Double): Double? {
    require(h > 0) { "grid step must be positive, got $h" }
    return if (eh is Measured.Reliable) constCh(eh.value, h, p) else null
}
