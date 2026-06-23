package numerics

import kotlin.math.exp
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals

class QuadratureTest {
    private val bp01 = doubleArrayOf(0.0, 1.0)

    @Test fun exactOnPolynomialsUpToDegree2mMinus1() {
        // m=4 nodes -> exact up to degree 7.
        val q = GaussLegendre(4)
        assertEquals(1.0 / 8.0, q.integrate(bp01) { t -> Math.pow(t, 7.0) }, 1e-12)
        assertEquals(1.0 / 7.0, q.integrate(bp01) { t -> Math.pow(t, 6.0) }, 1e-12)
    }

    @Test fun integralExpMinus1() {
        val q = GaussLegendre(8)
        assertEquals(exp(1.0) - 1.0, q.integrate(bp01) { t -> exp(t) }, 1e-12)
    }

    @Test fun integralOneOverOnePlusT_isLn2() {
        val q = GaussLegendre(8)
        assertEquals(ln(2.0), q.integrate(bp01) { t -> 1.0 / (1.0 + t) }, 1e-10)
    }

    @Test fun integrateIntervalMatchesBreakpoints() {
        val q = GaussLegendre(6)
        assertEquals(8.0 / 3.0, q.integrateInterval(0.0, 2.0) { t -> t * t }, 1e-10)
    }

    @Test fun compositeSubdivisionConsistent() {
        // Both single-interval and split quadratures approximate the same integral;
        // for a smooth integrand with 8 nodes/sub they must agree closely.
        val q = GaussLegendre(8)
        val analytic = (exp(2.0) - 1.0) / 2.0 // ∫_0^1 e^{2t} dt
        val whole = q.integrate(doubleArrayOf(0.0, 1.0)) { t -> exp(2.0 * t) }
        val split = q.integrate(doubleArrayOf(0.0, 0.3, 0.7, 1.0)) { t -> exp(2.0 * t) }
        assertEquals(analytic, whole, 1e-9)
        assertEquals(analytic, split, 1e-9)
    }
}
