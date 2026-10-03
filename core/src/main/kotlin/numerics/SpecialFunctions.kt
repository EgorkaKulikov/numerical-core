package numerics

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Special functions needed for weakly singular kernels and for exact solutions of model problems.
 *
 * Methods:
 * - [gamma], [lnGamma]: Stirling series with 8 terms (Bernoulli numbers up to `B_16`, truncation error below
 *   `2e-18` at `x ≥ 10`). [gamma] evaluates `√(2π)·e^S·x^((x−1/2)/2)·e^(−x)·x^((x−1/2)/2)` directly, not as
 *   `exp(lnGamma)`, whose relative error grows like `ulp(ln Γ(x))` (about `8e-14` at `x = 171`).
 *   Below 10 the recurrence `x·Γ(x) = Γ(x + k)/((x + 1)…(x + k − 1))` reaches `x + k ≥ 10`; the rounding errors of
 *   the shifted arguments are computed exactly (TwoSum) and compensated to first order, `Γ(y + e) ≈ Γ(y)(1 + ψ(y)e)`.
 *   `(n − 1)!` is multiplied out exactly for integer `n ≤ 23`.
 * - [beta]: `Γ(a)·(Γ(b)/Γ(a + b))` while `a + b ≤ 170`, with a first-order correction for the rounding of `a + b`;
 *   `exp(lnGamma(p) + lnGamma(q) − lnGamma(p + q))` above.
 * - [erfc]: Maclaurin series of `erf` for `|x| < 1`; for `x ≥ 1` the Legendre continued fraction of the
 *   incomplete gamma function `Γ(1/2, x²)` evaluated backward with a fixed number of terms, multiplied by
 *   `exp(−x²)` computed as `exp(−s²)·exp(−(x − s)(x + s))` with `s = ⌊16x⌋/16` (W. J. Cody's splitting:
 *   `s²` is exact, so the large exponent carries no rounding error); `erfc(x) = 2 − erfc(−x)` for `x ≤ −1`.
 * - [mittagLeffler]: the defining power series (positive terms, no cancellation).
 */
public object SpecialFunctions {
    private const val HALF_LN_2PI = 0.91893853320467274178
    private const val SQRT_2PI = 2.5066282746310005024
    private const val SQRT_PI = 1.7724538509055160273
    private const val TWO_OVER_SQRT_PI = 1.1283791670955125739

    /** Stirling coefficients `B_2k/(2k(2k − 1))`, `k = 1..8`. */
    private val STIRLING = doubleArrayOf(
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360360.0,
        1.0 / 156.0,
        -3617.0 / 122400.0,
    )

    /** Smallest argument at which the Stirling series is used directly. */
    private const val STIRLING_MIN = 10.0

    /** Γ(x) overflows `Double` for arguments above this bound. */
    private const val GAMMA_OVERFLOW = 171.625

    /** Largest integer argument for which `(n − 1)!` is computed exactly by repeated multiplication. */
    private const val EXACT_FACTORIAL_LIMIT = 23.0

    /** Largest `p + q` for which [beta] uses the gamma ratio (no overflow of the factors). */
    private const val BETA_RATIO_LIMIT = 170.0

    /** `erfc(x)` underflows to zero for `x` at or above this bound. */
    private const val ERFC_UNDERFLOW = 27.5

    /** Number of continued-fraction terms for `erfc` at `x²`: `CF_BASE + CF_SCALE/x²`. */
    private const val CF_BASE = 12
    private const val CF_SCALE = 160.0

    /** Relative size of the last series term at which summation stops. */
    private const val SERIES_TOLERANCE = 1e-17

    /**
     * Γ(x) for finite `x > 0` (relative error ≤ 1e-14 on `(0, 171]`); `+∞` where Γ(x) exceeds the `Double` range
     * (above about `171.62` and below about `5.6e-309`).
     *
     * @throws IllegalArgumentException if `x` is not finite and positive
     */
    public fun gamma(x: Double): Double {
        require(x > 0.0 && x.isFinite()) { "argument x must be finite and positive, got $x" }
        if (x > GAMMA_OVERFLOW) return Double.POSITIVE_INFINITY
        if (x <= EXACT_FACTORIAL_LIMIT && x == floor(x)) {
            var factorial = 1.0
            for (k in 2 until x.toInt()) factorial *= k
            return factorial
        }
        if (x >= STIRLING_MIN) return stirlingGamma(x)
        return shiftedGamma(x) / x
    }

    /**
     * ln Γ(x) for finite `x > 0`.
     *
     * @throws IllegalArgumentException if `x` is not finite and positive
     */
    public fun lnGamma(x: Double): Double {
        require(x > 0.0 && x.isFinite()) { "argument x must be finite and positive, got $x" }
        return lnGammaPositive(x)
    }

    private fun lnGammaPositive(x: Double): Double {
        if (x >= STIRLING_MIN) return (x - 0.5) * ln(x) - x + HALF_LN_2PI + stirlingSeries(x)
        return ln(shiftedGamma(x)) - ln(x)
    }

    /** `S(x) = ln Γ(x) − (x − 1/2) ln x + x − ln √(2π)` for `x ≥ 10`. */
    private fun stirlingSeries(x: Double): Double {
        val r = 1.0 / (x * x)
        var sum = STIRLING[STIRLING.size - 1]
        for (i in STIRLING.size - 2 downTo 0) sum = sum * r + STIRLING[i]
        return sum / x
    }

    /** Γ(x) for `x ≥ 10`; the power is split in two halves so that no intermediate overflows. */
    private fun stirlingGamma(x: Double): Double {
        val half = x.pow(0.5 * (x - 0.5))
        return SQRT_2PI * exp(stirlingSeries(x)) * half * exp(-x) * half
    }

    /** `x·Γ(x) = Γ(x + k)/((x + 1)…(x + k − 1))` for `0 < x < 10`, with `x + k ≥ 10`. */
    private fun shiftedGamma(x: Double): Double {
        var product = 1.0
        var relativeError = 0.0
        var j = 1.0
        var y = x + j
        while (y < STIRLING_MIN) {
            relativeError += twoSumError(x, j, y) / y
            product *= y
            j += 1.0
            y = x + j
        }
        val psi = ln(y) - 0.5 / y
        return stirlingGamma(y) * (1.0 + psi * twoSumError(x, j, y)) / (product * (1.0 + relativeError))
    }

    /** Exact rounding error `(a + b) − s` of `s = fl(a + b)` (Knuth's TwoSum). */
    private fun twoSumError(a: Double, b: Double, s: Double): Double {
        val bVirtual = s - a
        return (a - (s - bVirtual)) + (b - bVirtual)
    }

    /**
     * Euler beta function `B(p, q) = Γ(p)Γ(q)/Γ(p + q)` for finite `p, q > 0`.
     *
     * For `p + q ≤ 170` the gamma ratio is used, and the rounding error `e` of the floating-point sum
     * `s = p + q` is compensated by `Γ(s + e) ≈ Γ(s)·(1 + ψ(s)·e)` with `ψ(s) ≈ ln s − 1/(2s)`;
     * above, `exp(lnGamma(p) + lnGamma(q) − lnGamma(p + q))`, whose relative error is of the order of
     * the rounding unit of `lnGamma(p + q)`.
     *
     * @throws IllegalArgumentException if `p` or `q` is not finite and positive
     */
    public fun beta(p: Double, q: Double): Double {
        require(p > 0.0 && p.isFinite() && q > 0.0 && q.isFinite()) {
            "beta arguments p and q must be finite and positive, got p = $p, q = $q"
        }
        val small = min(p, q)
        val large = max(p, q)
        val s = small + large
        if (s > BETA_RATIO_LIMIT) return exp(lnGammaPositive(p) + lnGammaPositive(q) - lnGammaPositive(s))
        val sumError = small - (s - large)
        val psi = ln(s) - 0.5 / s
        return gamma(small) * (gamma(large) / gamma(s)) / (1.0 + psi * sumError)
    }

    /**
     * Complementary error function `erfc(x) = 1 − erf(x)`; relative error ≤ 1e-14 where `erfc(x) ≥ 1e-300`.
     * Returns `0` for `x ≥ 27.5` (underflow) and accepts infinite arguments: `erfc(+∞) = 0`, `erfc(−∞) = 2`.
     *
     * @throws IllegalArgumentException if `x` is NaN
     */
    public fun erfc(x: Double): Double {
        require(!x.isNaN()) { "argument x must not be NaN" }
        return when {
            x >= ERFC_UNDERFLOW -> 0.0
            x <= -1.0 -> 2.0 - erfc(-x)
            x < 1.0 -> 1.0 - TWO_OVER_SQRT_PI * erfSeries(x)
            else -> erfcContinuedFraction(x, continuedFractionTerms(x))
        }
    }

    /** `(√π/2)·erf(x) = Σ (−1)^n x^(2n+1) / (n!·(2n + 1))`, summed until the term is negligible. */
    private fun erfSeries(x: Double): Double {
        val xx = x * x
        var power = x
        var sum = x
        var n = 0
        while (true) {
            n++
            power *= -xx / n
            val term = power / (2 * n + 1)
            sum += term
            if (abs(term) <= SERIES_TOLERANCE * abs(sum)) return sum
        }
    }

    internal fun continuedFractionTerms(x: Double): Int = CF_BASE + (CF_SCALE / (x * x)).toInt()

    /**
     * `erfc(x)` for `x ≥ 1` from `Γ(1/2, X)/Γ(1/2) = (x·e^(−X)/√π) / (b0 + a1/(b1 + a2/(b2 + …)))`, `X = x²`,
     * `b_k = X + 1/2 + 2k`, `a_k = −k(k − 1/2)`, truncated after [terms] levels and evaluated backward.
     */
    internal fun erfcContinuedFraction(x: Double, terms: Int): Double {
        val xx = x * x
        var tail = xx + 0.5 + 2.0 * terms
        for (k in terms downTo 1) tail = xx + 0.5 + 2.0 * (k - 1) - k * (k - 0.5) / tail
        val split = floor(16.0 * x) / 16.0
        val gaussian = exp(-split * split) * exp(-(x - split) * (x + split))
        return x * gaussian / SQRT_PI / tail
    }

    /**
     * Mittag-Leffler function `E_β(z) = Σ_{k≥0} z^k / Γ(βk + 1)` for `1/3 ≤ β ≤ 2` and `0 ≤ z ≤ 10`:
     * positive-term series (no cancellation), terms via `exp(k ln z − lnGamma(βk + 1))`, summed until the
     * term is below 1e-17 of the sum (the terms are log-concave in `k`, so the stop is past the largest term).
     * Overflows to `+∞` where `E_β(z)` exceeds the `Double` range (e.g. `β = 1/3`, `z ≳ 8.9`).
     *
     * @throws IllegalArgumentException if `β` is outside `[1/3, 2]` or `z` is outside `[0, 10]` (or either is NaN)
     */
    public fun mittagLeffler(beta: Double, z: Double): Double {
        require(beta >= 1.0 / 3.0 && beta <= 2.0) { "Mittag-Leffler parameter beta must lie in [1/3, 2], got $beta" }
        require(z >= 0.0 && z <= 10.0) { "Mittag-Leffler argument z must lie in [0, 10], got $z" }
        if (z == 0.0) return 1.0
        val lnZ = ln(z)
        var sum = 1.0
        var k = 1
        while (true) {
            val term = exp(k * lnZ - lnGammaPositive(beta * k + 1.0))
            sum += term
            if (term < SERIES_TOLERANCE * sum) return sum
            k++
        }
    }
}
