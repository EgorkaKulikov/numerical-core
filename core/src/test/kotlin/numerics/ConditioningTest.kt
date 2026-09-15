package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Conditioning tests: the cond estimate via explicit inversion (with the inversion residual
 * as the reliability indicator) and spectral estimates via Jacobi rotations.
 *
 * The key requirement checked here: on a nearly singular matrix the API does not silently
 * return a meaningless number but flags the estimate as unreliable.
 */
@Tag("fast")
class ConditioningTest {

    /** ‖A‖∞ = maximum absolute row sum. */
    @Test fun matrixNormInfIsMaxRowSum() {
        val a = arrayOf(doubleArrayOf(1.0, -2.0), doubleArrayOf(3.0, 4.0))
        assertEquals(7.0, Conditioning.matrixNormInf(a), 1e-15)
    }

    /** An empty matrix is a contract violation, not 0.0. */
    @Test fun matrixNormInfRejectsEmpty() {
        assertFailsWith<IllegalArgumentException> { Conditioning.matrixNormInf(arrayOf()) }
        assertFailsWith<IllegalArgumentException> { Conditioning.matrixNormInf(arrayOf(doubleArrayOf())) }
    }

    /** 2x2 inversion: checked against the closed-form inverse. */
    @Test fun inverseMatchesClosedForm() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val inv = Conditioning.inverse(a)!!
        // A^{-1} = 1/det * [[d, -b], [-c, a]], det = -2
        assertEquals(-2.0, inv[0][0], 1e-12)
        assertEquals(1.0, inv[0][1], 1e-12)
        assertEquals(1.5, inv[1][0], 1e-12)
        assertEquals(-0.5, inv[1][1], 1e-12)
    }

    /** Non-square and empty matrices are rejected by the inversion contract. */
    @Test fun inverseRejectsMalformed() {
        assertFailsWith<IllegalArgumentException> { Conditioning.inverse(arrayOf()) }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.inverse(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0)))
        }
    }

    /** cond∞ of a 2x2 matches the analytic value ‖A‖∞·‖A^{-1}‖∞ = 7*3 = 21. */
    @Test fun conditionInfMatchesAnalyticValue() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val est = Conditioning.conditionInf(a)
        assertTrue(est.isReliable, "a well-conditioned matrix must yield a reliable estimate")
        assertEquals(21.0, est.condInf, 1e-10)
        assertEquals(21.0, est.valueOrNull()!!, 1e-10)
        assertTrue(est.inversionResidual < 1e-14, "inversion residual = ${est.inversionResidual}")
    }

    /** cond∞(I) = 1 — the lower bound of the condition number is attained. */
    @Test fun conditionInfOfIdentityIsOne() {
        val est = Conditioning.conditionInf(LinearAlgebra.identity(5))
        assertTrue(est.isReliable)
        assertEquals(1.0, est.condInf, 1e-12)
    }

    /** Singular matrix: the estimate is infinite, flagged unreliable, valueOrNull = null. */
    @Test fun conditionInfReportsSingularAsUnreliable() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0))
        val est = Conditioning.conditionInf(a)
        assertTrue(!est.isReliable, "a singular matrix cannot yield a reliable estimate")
        assertNull(est.valueOrNull())
        assertNull(Conditioning.inverse(a))
    }

    /** A non-positive reliability tolerance is a contract violation: it would mean no check at all. */
    @Test fun conditionInfRejectsNonPositiveTolerance() {
        assertFailsWith<IllegalArgumentException> {
            Conditioning.conditionInf(LinearAlgebra.identity(2), tolerance = 0.0)
        }
    }

    /** Eigenvalues of the symmetric 2x2 [[2,1],[1,2]] are 1 and 3, in ascending order. */
    @Test fun symmetricEigenvaluesOfTwoByTwo() {
        val a = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 2.0))
        val eig = Conditioning.symmetricEigenvalues(a)
        assertEquals(1.0, eig[0], 1e-12)
        assertEquals(3.0, eig[1], 1e-12)
        assertEquals(1.0, Conditioning.smallestMagnitudeEigenvalue(a), 1e-12)
        assertEquals(3.0, Conditioning.conditionSymmetric(a), 1e-12)
    }

    /** A diagonal matrix is already diagonal: the method must exit immediately without disturbing the spectrum. */
    @Test fun symmetricEigenvaluesOfDiagonalAreDiagonalEntries() {
        val a = arrayOf(
            doubleArrayOf(3.0, 0.0, 0.0),
            doubleArrayOf(0.0, -1.0, 0.0),
            doubleArrayOf(0.0, 0.0, 2.0),
        )
        val eig = Conditioning.symmetricEigenvalues(a)
        assertEquals(-1.0, eig[0], 1e-14)
        assertEquals(2.0, eig[1], 1e-14)
        assertEquals(3.0, eig[2], 1e-14)
        // min|lambda| = 1, max|lambda| = 3
        assertEquals(3.0, Conditioning.conditionSymmetric(a), 1e-12)
    }

    /** Zero off-diagonal entries are skipped without a rotation: 3x3 coupled only through (0,2). */
    @Test fun symmetricEigenvaluesSkipsZeroOffDiagonal() {
        val a = arrayOf(
            doubleArrayOf(2.0, 0.0, 1.0),
            doubleArrayOf(0.0, 5.0, 0.0),
            doubleArrayOf(1.0, 0.0, 2.0),
        )
        val eig = Conditioning.symmetricEigenvalues(a)
        assertEquals(1.0, eig[0], 1e-12)
        assertEquals(3.0, eig[1], 1e-12)
        assertEquals(5.0, eig[2], 1e-12)
    }

    /** The eigenvalues sum to the trace — an independent check on a non-trivial spectrum. */
    @Test fun symmetricEigenvaluesPreserveTrace() {
        val n = 6
        val a = Array(n) { i -> DoubleArray(n) { j -> 1.0 / (1.0 + i + j) } }
        val eig = Conditioning.symmetricEigenvalues(a)
        var trace = 0.0
        for (i in 0 until n) trace += a[i][i]
        assertEquals(trace, eig.sum(), 1e-10)
    }

    /** Non-symmetric, empty and non-square matrices are rejected by the contract. */
    @Test fun symmetricEigenvaluesRejectMalformed() {
        assertFailsWith<IllegalArgumentException> {
            Conditioning.symmetricEigenvalues(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 1.0)))
        }
        assertFailsWith<IllegalArgumentException> { Conditioning.symmetricEigenvalues(arrayOf()) }
        assertFailsWith<IllegalArgumentException> {
            Conditioning.symmetricEigenvalues(arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(1.0, 2.0, 3.0)))
        }
    }

    /** Symmetric singular matrix: cond2 = +Inf, not some large random number. */
    @Test fun conditionSymmetricOfSingularIsInfinite() {
        val a = arrayOf(doubleArrayOf(1.0, 1.0), doubleArrayOf(1.0, 1.0))
        assertEquals(0.0, Conditioning.smallestMagnitudeEigenvalue(a), 1e-14)
        assertTrue(Conditioning.conditionSymmetric(a).isInfinite())
    }

    /**
     * Applicability boundary, part 1 (reliable regime): on a well-conditioned symmetric
     * matrix the inversion-based estimate and the Jacobi method agree.
     *
     * For symmetric A the norms ‖·‖∞ and ‖·‖2 differ by at most a factor of n,
     * so we compare not the numbers themselves but that both estimates lie in the same
     * corridor (and both are finite, and both >= 1).
     */
    @Test fun wellConditionedMatrixAgreesBetweenMethods() {
        val n = 16
        // A = I + 0.1 * symmetric smooth kernel -> diagonally dominant, cond ~ 1.
        val a = Array(n) { i -> DoubleArray(n) { j -> (if (i == j) 1.0 else 0.0) + 0.1 / (1.0 + abs(i - j)) } }
        val est = Conditioning.conditionInf(a)
        val cond2 = Conditioning.conditionSymmetric(a)
        assertTrue(est.isReliable, "inversion residual = ${est.inversionResidual}")
        assertTrue(cond2.isFinite() && cond2 >= 1.0)
        assertTrue(est.condInf >= 1.0)
        assertTrue(est.condInf <= n * cond2 && cond2 <= n * est.condInf, "cond_inf=${est.condInf}, cond_2=$cond2")
    }

    /**
     * Applicability boundary, part 2 (unreliable regime): the matrix with entries
     * 1/(1 + (i + j)/n) (Cauchy-like) is numerically singular (cond ≫ 1/ε).
     *
     * The Jacobi method gives sigma_min = 0, while the inversion-based estimate is either
     * declared unreliable by its residual or infinite. This contract closes the class of
     * bugs "noise of order 1e16...1e19 printed as a condition number".
     */
    @Test fun nearlySingularCauchyMatrixIsReportedUnreliable() {
        val n = 32
        val a = Array(n) { i ->
            DoubleArray(n) { j -> 1.0 / (1.0 + i.toDouble() / n + j.toDouble() / n) }
        }
        val sigmaMin = Conditioning.smallestMagnitudeEigenvalue(a)
        assertTrue(sigmaMin < 1e-14 * Conditioning.matrixNormInf(a), "sigma_min = $sigmaMin")
        assertTrue(Conditioning.conditionSymmetric(a) > 1e14)
        val est = Conditioning.conditionInf(a)
        assertTrue(
            !est.isReliable,
            "the inversion-based estimate must be flagged unreliable: cond=${est.condInf}, residual=${est.inversionResidual}",
        )
        assertNull(est.valueOrNull())
    }

    /**
     * A diagonal shift brings the same matrix back into the reliable regime: after adding
     * alpha·I with alpha = 1e-2 the inversion residual drops below the tolerance and the estimate becomes usable.
     */
    @Test fun diagonalShiftRestoresReliability() {
        val n = 32
        val a = Array(n) { i ->
            DoubleArray(n) { j ->
                (if (i == j) 1e-2 else 0.0) + 1.0 / (1.0 + i.toDouble() / n + j.toDouble() / n)
            }
        }
        val est = Conditioning.conditionInf(a)
        assertTrue(est.isReliable, "inversion residual = ${est.inversionResidual}")
        assertTrue(est.condInf > 1.0 && est.condInf.isFinite())
    }
}
