package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests of the forward-error diagnostics for linear system solutions.
 *
 * Requirement under test: a small backward error (the only thing the `solve` post-check
 * controls, via the [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE] threshold) must not
 * look like evidence of an accurate result. On numerically singular input the API must
 * either honestly say "there is no bound" or flag the estimate as unreliable, but never
 * return a number that looks like the truth.
 */
@Tag("fast")
class ForwardErrorTest {

    // --- Backward error: a measurement, not an estimate -----------------------

    /** On the exact solution the residual is zero, hence the relative backward error is 0. */
    @Test fun backwardErrorOfExactSolutionIsZero() {
        val a = arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 4.0))
        val b = doubleArrayOf(2.0, 4.0)
        assertEquals(0.0, Conditioning.relativeBackwardError(a, b, doubleArrayOf(1.0, 1.0)), 0.0)
    }

    /**
     * The value matches the formula `‖Ax−b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)` — the same one
     * used in the `solve` post-check. Here ‖A‖∞ = 4, ‖x‖∞ = 1, ‖b‖∞ = 4,
     * ‖Ax−b‖∞ = |2·1 − 3| = 1, so the answer is 1/4.
     */
    @Test fun backwardErrorMatchesSolveNormalisation() {
        val a = arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 4.0))
        val b = doubleArrayOf(3.0, 4.0)
        assertEquals(0.25, Conditioning.relativeBackwardError(a, b, doubleArrayOf(1.0, 1.0)), 1e-15)
    }

    /** Zero scale (A = 0, b = 0) gives 0.0, not NaN from dividing zero by zero. */
    @Test fun backwardErrorOfZeroSystemIsZeroNotNaN() {
        val a = arrayOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0, 0.0))
        val z = doubleArrayOf(0.0, 0.0)
        assertEquals(0.0, Conditioning.relativeBackwardError(a, z, z), 0.0)
    }

    /** Empty, non-square and length-mismatched inputs are rejected by the contract. */
    @Test fun backwardErrorRejectsMalformed() {
        val a = arrayOf(doubleArrayOf(1.0, 0.0), doubleArrayOf(0.0, 1.0))
        val v = doubleArrayOf(1.0, 1.0)
        assertFailsWith<IllegalArgumentException> {
            Conditioning.relativeBackwardError(arrayOf(), DoubleArray(0), DoubleArray(0))
        }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.relativeBackwardError(arrayOf(doubleArrayOf(1.0, 2.0, 3.0)), doubleArrayOf(1.0), doubleArrayOf(1.0))
        }
        assertFailsWith<IllegalArgumentException> { Conditioning.relativeBackwardError(a, doubleArrayOf(1.0), v) }
        assertFailsWith<IllegalArgumentException> { Conditioning.relativeBackwardError(a, v, doubleArrayOf(1.0)) }
    }

    // --- The three ForwardError modes ----------------------------------------

    /** A reliable cond estimate gives the bound `cond · ω` and the number of surviving digits. */
    @Test fun reliableConditionGivesBoundedForwardError() {
        val est = ConditionEstimate(1e6, 1e-15, Conditioning.INVERSION_RESIDUAL_TOLERANCE)
        val fe = Conditioning.forwardError(est, 1e-16)
        assertTrue(fe is ForwardError.Bounded)
        assertEquals(1e-10, fe.relativeBound, 1e-24)
        assertEquals(1e-10, fe.relativeBoundOrNull()!!, 1e-24)
        assertEquals(1e-16, fe.backwardError, 0.0)
        assertEquals(10.0, fe.survivingDigitsOrNull()!!, 1e-12)
    }

    /**
     * An unreliable cond estimate is not turned into a number: mode [ForwardError.Unreliable],
     * `relativeBoundOrNull() == null`. This is precisely the guard against "a number that looks like the truth".
     */
    @Test fun unreliableConditionYieldsNoNumber() {
        val est = ConditionEstimate(3.7e18, 760.0, Conditioning.INVERSION_RESIDUAL_TOLERANCE)
        val fe = Conditioning.forwardError(est, 1e-16)
        assertTrue(fe is ForwardError.Unreliable)
        assertNull(fe.relativeBoundOrNull())
        assertNull(fe.survivingDigitsOrNull())
        // The unreliable estimate itself is kept for flagged printing and root-cause analysis.
        assertEquals(760.0, fe.condition.inversionResidual, 0.0)
    }

    /**
     * "No finite cond at all" differs from "cond is large but finite":
     * an infinite estimate gives [ForwardError.NoFiniteBound], not [ForwardError.Unreliable].
     */
    @Test fun infiniteConditionIsDistinctFromUnreliable() {
        val est = ConditionEstimate(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, 1e-8)
        val fe = Conditioning.forwardError(est, 1e-16)
        assertTrue(fe is ForwardError.NoFiniteBound)
        assertNull(fe.relativeBoundOrNull())
        assertEquals(1e-16, fe.backwardError, 0.0)
    }

    /** Zero backward error means the full mantissa: 16 digits, not infinity. */
    @Test fun zeroBackwardErrorGivesFullMantissa() {
        val est = ConditionEstimate(21.0, 0.0, 1e-8)
        val fe = Conditioning.forwardError(est, 0.0)
        assertEquals(0.0, fe.relativeBoundOrNull()!!, 0.0)
        assertEquals(16.0, fe.survivingDigitsOrNull()!!, 0.0)
    }

    /** A bound >= 1 means "no guaranteed correct digits at all", not a negative number. */
    @Test fun boundAboveOneMeansZeroSurvivingDigits() {
        val est = ConditionEstimate(1e14, 1e-15, 1e-8)
        val fe = Conditioning.forwardError(est, 1e-13)
        assertTrue(fe.relativeBoundOrNull()!! > 1.0)
        assertEquals(0.0, fe.survivingDigitsOrNull()!!, 0.0)
    }

    /** A non-finite or negative backward error is a contract violation: it is not a measurement. */
    @Test fun forwardErrorRejectsNonMeasuredBackwardError() {
        val est = ConditionEstimate(21.0, 0.0, 1e-8)
        assertFailsWith<IllegalArgumentException> { Conditioning.forwardError(est, Double.NaN) }
        assertFailsWith<IllegalArgumentException> { Conditioning.forwardError(est, -1e-16) }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.forwardErrorSymmetric(LinearAlgebra.identity(2), Double.POSITIVE_INFINITY)
        }
    }

    // --- Spectral path: sigma_min = 0 is distinguishable from "cond is large but finite" ---

    /**
     * An exactly singular symmetric matrix: the spectral path honestly says
     * "there is no finite bound" instead of producing a large random number.
     */
    @Test fun symmetricPathReportsNoFiniteBoundOnSingularMatrix() {
        val a = arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 1.0))
        assertEquals(0.0, Conditioning.smallestMagnitudeEigenvalue(a), 1e-14)
        val fe = Conditioning.forwardErrorSymmetric(a, 1e-16)
        assertTrue(fe is ForwardError.NoFiniteBound)
        assertNull(fe.relativeBoundOrNull())
    }

    /** On a well-conditioned symmetric matrix the spectral path gives a finite bound. */
    @Test fun symmetricPathBoundsWellConditionedMatrix() {
        val a = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 2.0))
        val fe = Conditioning.forwardErrorSymmetric(a, 1e-16)
        assertTrue(fe is ForwardError.Bounded)
        // cond2 = 3/1 = 3, hence the bound is 3e-16.
        assertEquals(3.0, fe.cond, 1e-12)
        assertEquals(3e-16, fe.relativeBound, 1e-28)
    }

    /** A non-symmetric matrix on the spectral path is a contract violation, not a silently wrong answer. */
    @Test fun symmetricPathRejectsAsymmetricMatrix() {
        assertFailsWith<IllegalArgumentException> {
            Conditioning.forwardErrorSymmetric(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 1.0)), 1e-16)
        }
    }

    // --- solveDiagnosed ------------------------------------------------------

    /**
     * The solution from [LinearAlgebra.solveDiagnosed] is bitwise identical to that of
     * [LinearAlgebra.solve]: the diagnostics neither refine nor alter anything.
     */
    @Test fun diagnosedSolutionIsBitwiseIdenticalToSolve() {
        val n = 16
        val a = Array(n) { i -> DoubleArray(n) { j -> (if (i == j) 1.0 else 0.0) + 0.1 / (1.0 + abs(i - j)) } }
        val b = DoubleArray(n) { 1.0 }
        val plain = LinearAlgebra.solve(a, b)
        val diagnosed = LinearAlgebra.solveDiagnosed(a, b)
        for (i in 0 until n) {
            assertEquals(plain[i].toRawBits(), diagnosed.x[i].toRawBits(), "component $i differs")
        }
    }

    /**
     * A well-conditioned system: the forward-error bound is finite and small,
     * and almost the whole mantissa of significant digits survives.
     *
     * Reference values recorded from the current run: `cond∞ ≈ 1.669`, bound ~5.1e-16,
     * i.e. more than fifteen correct decimal digits.
     */
    @Test fun wellConditionedSystemKeepsAlmostAllDigits() {
        val n = 16
        val a = Array(n) { i -> DoubleArray(n) { j -> (if (i == j) 1.0 else 0.0) + 0.1 / (1.0 + abs(i - j)) } }
        val b = DoubleArray(n) { 1.0 }
        val fe = LinearAlgebra.solveDiagnosed(a, b).forwardError
        assertTrue(fe is ForwardError.Bounded, "a well-conditioned system must yield a bound, got $fe")
        assertTrue(fe.cond < 2.0, "cond=${fe.cond}")
        assertTrue(fe.relativeBound < 1e-14, "bound=${fe.relativeBound}")
        assertTrue(fe.survivingDigitsOrNull()!! > 14.0, "digits=${fe.survivingDigitsOrNull()}")
    }

    /**
     * A numerically singular matrix (cond ≫ 1/ε), part 1: a matrix with entries
     * `1/(1 + (i + j)/n)` (Cauchy-like). No path may return a small forward-error
     * bound, however small the backward error is.
     *
     * The backward error is deliberately fixed (machine epsilon) rather than obtained
     * from `solve`: on such input the backends behave differently — `reference` and
     * native LAPACK may declare the matrix singular at different steps, and both
     * answers are honest (see the applicability limits in the KDoc of
     * [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE]). What is tested here is not the
     * backend's behaviour but the diagnostics', and that must be identical everywhere.
     */
    @Test fun nearlySingularCauchyMatrixNeverYieldsSmallForwardBound() {
        val n = 32
        val a = Array(n) { i -> DoubleArray(n) { j -> 1.0 / (1.0 + i.toDouble() / n + j.toDouble() / n) } }
        val omega = 2.220446049250313e-16 // machine epsilon: the best any solver can achieve

        // Inversion path: no number — either the estimate is unreliable or cond is infinite.
        val viaInversion = Conditioning.forwardError(Conditioning.conditionInf(a), omega)
        assertNull(viaInversion.relativeBoundOrNull(), "got $viaInversion")
        assertNull(viaInversion.survivingDigitsOrNull())
        assertTrue(
            viaInversion is ForwardError.Unreliable || viaInversion is ForwardError.NoFiniteBound,
            "got $viaInversion",
        )

        // The spectral path is deterministic (Jacobi does not depend on the backend): a bound
        // exists, but it exceeds one — not a single correct digit remains.
        val viaSpectrum = Conditioning.forwardErrorSymmetric(a, omega)
        assertTrue(viaSpectrum is ForwardError.Bounded, "got $viaSpectrum")
        assertTrue(viaSpectrum.relativeBound > 1.0, "bound=${viaSpectrum.relativeBound}")
        assertEquals(0.0, viaSpectrum.survivingDigitsOrNull()!!, 0.0)
    }

    /**
     * A numerically singular matrix, part 2: a small backward error does not imply
     * an accurate result — on a system that every backend solves.
     *
     * The matrix `diag(1e-9, 1, …, 1)` is non-singular and ill-conditioned:
     * `cond∞ = ‖1‖ · ‖1e9‖ = 1e9`. The diagonal form is chosen deliberately: all
     * quantities are known analytically and identical on every backend, so the test
     * checks precisely the diagnostics logic, not the arithmetic of a particular LU.
     *
     * `solve` issues no warning on this system — rightly so: the backward error is negligible.
     * But it cannot guarantee the accuracy of the result: as soon as the backward error
     * is at the level of machine epsilon (the best any solver can achieve at all),
     * the factor `cond∞ = 1e9` eats nine of the sixteen decimal digits. It is exactly
     * this factor that was invisible in the output of `solve`.
     */
    @Test fun smallBackwardErrorDoesNotImplyAccurateResult() {
        val n = 6
        val a = Array(n) { i -> DoubleArray(n) { j -> if (i == j) (if (i == 0) 1e-9 else 1.0) else 0.0 } }
        val b = DoubleArray(n) { 1.0 }

        val d = LinearAlgebra.solveDiagnosed(a, b)
        val fe = d.forwardError
        // solve stays silent: the backward error is negligible — the system is solved backward-stably.
        assertTrue(fe.backwardError < 1e-14, "ω=${fe.backwardError}")

        // The cond estimate is reliable (a diagonal matrix is inverted exactly) and large.
        assertTrue(fe is ForwardError.Bounded, "got $fe")
        assertEquals(1e9, fe.cond, 1e-3)

        // The point: for any realistic backward error the cond factor eats 9 digits.
        val atMachineEpsilon = Conditioning.forwardError(
            ConditionEstimate(fe.cond, 0.0, Conditioning.INVERSION_RESIDUAL_TOLERANCE),
            2.220446049250313e-16,
        )
        val digits = atMachineEpsilon.survivingDigitsOrNull()!!
        assertTrue(digits in 6.0..8.0, "guaranteed digits $digits, expected about 7 instead of 16")
    }

    /**
     * The singularity contract is not weakened: on an exactly singular matrix
     * [LinearAlgebra.solveDiagnosed] throws just like [LinearAlgebra.solve],
     * and the diagnostics are never reached.
     */
    @Test fun diagnosedSolveKeepsSingularityContract() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0))
        val b = doubleArrayOf(1.0, 3.0)
        assertFailsWith<IllegalStateException> { LinearAlgebra.solve(a, b) }
        assertFailsWith<IllegalStateException> { LinearAlgebra.solveDiagnosed(a, b) }
    }

    /** The reliability tolerance is forwarded: an overly tight tolerance turns a reliable estimate into an unreliable one. */
    @Test fun toleranceIsForwardedToConditionEstimate() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val b = doubleArrayOf(1.0, 1.0)
        assertTrue(LinearAlgebra.solveDiagnosed(a, b).forwardError is ForwardError.Bounded)
        val strict = LinearAlgebra.solveDiagnosed(a, b, tolerance = Double.MIN_VALUE).forwardError
        assertTrue(strict is ForwardError.Unreliable, "got $strict")
    }
}
