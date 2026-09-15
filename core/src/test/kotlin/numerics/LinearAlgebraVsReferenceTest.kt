package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Cross-check of the [numerics.backend.Backends.java] and (when available)
 * [numerics.backend.Backends.native] implementations through the single entry point [LinearAlgebra]
 * against the independent oracle [ReferenceOracle] on seeded data with tolerance 1e-8.
 */
@Tag("fast")
@Suppress("DEPRECATION")
class LinearAlgebraVsReferenceTest {

    private val tol = 1e-8
    private val sizes = intArrayOf(3, 8, 20)
    private val backends: List<LinAlgBackend> = Backends.available()

    /** One DynamicTest per (backend, size) pair: a failure of one case does not hide the others. */
    private fun perBackendAndSize(name: String, body: (LinAlgBackend, Int) -> Unit): List<DynamicTest> =
        backends.flatMap { backend ->
            sizes.map { n -> DynamicTest.dynamicTest("$name [${backend.name}, n=$n]") { body(backend, n) } }
        }

    private fun randMatrix(rnd: Random, rows: Int, cols: Int): Array<DoubleArray> =
        Array(rows) { DoubleArray(cols) { rnd.nextDouble(-1.0, 1.0) } }

    private fun randVector(rnd: Random, n: Int): DoubleArray =
        DoubleArray(n) { rnd.nextDouble(-1.0, 1.0) }

    /** Diagonally dominant (hence non-singular) n x n matrix. */
    private fun diagDominant(rnd: Random, n: Int): Array<DoubleArray> {
        val a = randMatrix(rnd, n, n)
        for (i in 0 until n) {
            var rowSum = 0.0
            for (j in 0 until n) rowSum += kotlin.math.abs(a[i][j])
            a[i][i] += rowSum + 1.0
        }
        return a
    }

    private fun assertMatEq(expected: Array<DoubleArray>, actual: Array<DoubleArray>) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            assertEquals(expected[i].size, actual[i].size)
            for (j in expected[i].indices) {
                assertTrue(
                    kotlin.math.abs(expected[i][j] - actual[i][j]) < tol,
                    "mismatch at [$i][$j]: ${expected[i][j]} vs ${actual[i][j]}"
                )
            }
        }
    }

    private fun assertVecEq(expected: DoubleArray, actual: DoubleArray) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            assertTrue(
                kotlin.math.abs(expected[i] - actual[i]) < tol,
                "mismatch at [$i]: ${expected[i]} vs ${actual[i]}"
            )
        }
    }

    /** Backend matVec matches the reference on sizes 3, 8, 20. */
    @TestFactory
    fun matVecMatchesReference() = perBackendAndSize("matVecMatchesReference") { backend, n ->
        val rnd = Random(1000 + n)
        val a = randMatrix(rnd, n, n)
        val x = randVector(rnd, n)
        assertVecEq(ReferenceOracle.matVec(a, x), LinearAlgebra.matVec(a, x, backend))
    }

    /** Backend matTransVec matches the reference (rectangular matrices). */
    @TestFactory
    fun matTransVecMatchesReference() = perBackendAndSize("matTransVecMatchesReference") { backend, n ->
        val rnd = Random(2000 + n)
        val a = randMatrix(rnd, n, n + 2)
        val y = randVector(rnd, n)
        assertVecEq(ReferenceOracle.matTransVec(a, y), LinearAlgebra.matTransVec(a, y, backend))
    }

    /** Backend matMat matches the reference on rectangular factors. */
    @TestFactory
    fun matMatMatchesReference() = perBackendAndSize("matMatMatchesReference") { backend, n ->
        val rnd = Random(3000 + n)
        val a = randMatrix(rnd, n, n + 1)
        val b = randMatrix(rnd, n + 1, n + 3)
        assertMatEq(ReferenceOracle.matMat(a, b), LinearAlgebra.matMat(a, b, backend))
    }

    /** Backend atWa (A^T diag(w) A) matches the reference. */
    @TestFactory
    fun atWaMatchesReference() = perBackendAndSize("atWaMatchesReference") { backend, n ->
        val rnd = Random(4000 + n)
        val a = randMatrix(rnd, n + 2, n)
        val w = randVector(rnd, n + 2)
        assertMatEq(ReferenceOracle.atWa(a, w), LinearAlgebra.atWa(a, w, backend))
    }

    /** Backend addScaled (A + s*B) matches the reference. */
    @TestFactory
    fun addScaledMatchesReference() = perBackendAndSize("addScaledMatchesReference") { backend, n ->
        val rnd = Random(5000 + n)
        val a = randMatrix(rnd, n, n)
        val b = randMatrix(rnd, n, n)
        val s = rnd.nextDouble(-2.0, 2.0)
        assertMatEq(ReferenceOracle.addScaled(a, b, s), LinearAlgebra.addScaled(a, b, s, backend))
    }

    /** Backend solve matches the reference on well-conditioned systems. */
    @TestFactory
    fun solveMatchesReference() = perBackendAndSize("solveMatchesReference") { backend, n ->
        val rnd = Random(6000 + n)
        val a = diagDominant(rnd, n)
        val b = randVector(rnd, n)
        assertVecEq(ReferenceOracle.solve(a, b), LinearAlgebra.solve(a, b, backend))
    }

    /**
     * Uniform singularity semantics: on an exactly singular system of any scale both
     * backends must throw an exception of the same type through the single entry point.
     *
     * Runs on all available backends (`-Dnumerics.backend=native|java`).
     *
     * On the choice of cases. These are exactly singular matrices (rank < n), not
     * ill-conditioned ones (such as the Hilbert matrix): ill-conditioning by itself is
     * not singularity — ill-conditioned but non-singular matrices (cond ~ 1e10) are solved
     * with a small residual, and demanding the same reaction to them from the backends
     * would be wrong (see the KDoc of [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE]).
     */
    @TestFactory
    fun bothBackendsRejectSingularSystemsAtAnyScale(): List<DynamicTest> {
        val cases = mutableListOf<DynamicTest>()
        // (a) Rank 1 at different scales — scale invariance of the semantics.
        for (scale in doubleArrayOf(1e-8, 1.0, 1e8)) {
            cases += DynamicTest.dynamicTest("rank 1, scale=$scale") {
                val a = arrayOf(
                    doubleArrayOf(1.0 * scale, 2.0 * scale),
                    doubleArrayOf(2.0 * scale, 4.0 * scale),
                )
                val b = doubleArrayOf(1.0 * scale, 3.0 * scale)
                assertFailsWith<IllegalStateException>("scale=$scale, backend ${Backends.default().name}") {
                    LinearAlgebra.solve(a, b)
                }
            }
        }
        cases += DynamicTest.dynamicTest("singularity by rotation and rank 2 in size 3") {
        // (b) Singularity by rotation: diag(1, 0) in a basis rotated by 45 degrees —
        // no matrix entry is small, yet the matrix is singular.
        val rotated = arrayOf(doubleArrayOf(0.5, 0.5), doubleArrayOf(0.5, 0.5))
        assertFailsWith<IllegalStateException>("backend ${Backends.default().name}") {
            LinearAlgebra.solve(rotated, doubleArrayOf(1.0, 0.0))
        }
        // (c) Singularity in a larger size: the second row is exactly twice the first.
        // All numbers are powers of two, so the singularity is exact in IEEE-754 and
        // any LU sees it (the pivot becomes exactly zero).
        val rank2 = arrayOf(
            doubleArrayOf(1.0, 2.0, 4.0),
            doubleArrayOf(2.0, 4.0, 8.0),
            doubleArrayOf(1.0, 4.0, 16.0),
        )
        assertFailsWith<IllegalStateException>("backend ${Backends.default().name}") {
            LinearAlgebra.solve(rank2, doubleArrayOf(1.0, 3.0, 1.0))
        }
        }
        return cases
    }

    /**
     * A contract boundary discovered empirically and recorded here as deliberate:
     * the residual criterion does not catch an inconsistent nearly singular system
     * if the backend returned a solution of very large norm.
     *
     * The reason is fundamental, not a defect of the threshold: with `||x|| ~ 1e16` a residual
     * of order `||b||` is still small relative to `||A||*||x||`, i.e. such an x exactly solves
     * a nearby system (small backward error) — formally an honest answer. It could only be
     * rejected by a conditioning check, which is forbidden by the neutrality requirement
     * (ill-conditioned but non-singular matrices with cond ~ 1e10 must be solved). The test
     * pins down exactly what is guaranteed: any backend either throws [IllegalStateException]
     * or returns a finite solution with a small backward error — but never returns NaN/Inf
     * and never lies about the residual.
     */
    @Test
    fun inconsistentRankDeficientSystemEitherThrowsOrHasSmallBackwardError() {
        // The third row equals the sum of the first two, but the right-hand side does not (4 != 1+2).
        val a = arrayOf(
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
            doubleArrayOf(5.0, 7.0, 9.0),
        )
        val b = doubleArrayOf(1.0, 2.0, 4.0)
        val x: DoubleArray? = try {
            LinearAlgebra.solve(a, b)
        } catch (e: IllegalStateException) {
            // An honest failure is also an acceptable outcome of the contract (this is how the
            // oracle's hand-written LU behaves), but it must be informative: the message names the cause.
            // Without this check the test would be a tautology on the reference backend.
            assertTrue(
                e.message?.contains("singular") == true,
                "a failure must name the cause: ${e.message}",
            )
            null
        }
        if (x != null) {
            for (v in x) assertTrue(v.isFinite(), "NaN/Inf is forbidden by the contract, got $v")
            // If a solution was returned after all, it must have a small backward error:
            // that is exactly what the single entry point checked before letting it through. The norm
            // ||A||_inf is computed from the matrix itself, not a literal: an inflated constant would make
            // the bound weaker than the one already checked in the entry point, and the test would catch nothing new.
            val matrixNormInf = (0..2).maxOf { i -> (0..2).sumOf { j -> kotlin.math.abs(a[i][j]) } }
            val residual = LinearAlgebra.normInf(
                DoubleArray(3) { i -> (0..2).sumOf { j -> a[i][j] * x[j] } - b[i] }
            )
            val bound = LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE *
                maxOf(matrixNormInf * LinearAlgebra.normInf(x), LinearAlgebra.normInf(b))
            assertTrue(residual <= bound, "residual $residual must be <= $bound")
        }
    }

    /**
     * An ill-conditioned but non-singular system (8x8 Hilbert matrix, cond ~ 1e10)
     * is not rejected by the residual threshold in the single entry point.
     *
     * This guards against regression: ill-conditioned but non-singular matrices with
     * such a condition number must be solved, and tightening the threshold would reject
     * them instead of a non-finite result.
     */
    @Test
    fun illConditionedButSolvableSystemIsAccepted() {
        val n = 8
        val hilbert = Array(n) { i -> DoubleArray(n) { j -> 1.0 / (i + j + 1) } }
        val rhs = DoubleArray(n) { 1.0 }
        val x = LinearAlgebra.solve(hilbert, rhs)
        // Actual check that a solution was returned at all and that it is finite.
        assertEquals(n, x.size)
        for (v in x) assertTrue(v.isFinite(), "solution must be finite, got $v")
    }
}
