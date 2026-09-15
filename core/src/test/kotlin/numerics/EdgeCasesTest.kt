package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.backend.MatrixNorm
import numerics.golden.GoldenInputs
import org.junit.jupiter.api.Assertions.assertTimeoutPreemptively
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.DynamicTest.dynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import org.junit.jupiter.api.function.ThrowingSupplier
import java.time.Duration
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Edge cases of the public API: backend registry, 1×1 matrices, rectangular and empty
 * matrices, large sizes, singularity of different structure, numerically extreme scales,
 * extreme quadrature and parallel assembly parameters, boundaries of [measured].
 *
 * Each case is a separate DynamicTest; linear algebra operations run on both
 * backends (the native one via `assumeTrue(isNativeAvailable())`).
 */
@Tag("fast")
class EdgeCasesTest {

    private fun m(vararg rows: DoubleArray): DenseMatrix = DenseMatrix.fromRows(arrayOf(*rows))
    private fun v(vararg x: Double): DoubleArray = doubleArrayOf(*x)

    private fun assertVec(expected: DoubleArray, actual: DoubleArray, tol: Double = 1e-12, what: String = "") {
        assertEquals(expected.size, actual.size, "$what: length")
        for (i in expected.indices) assertTrue(abs(expected[i] - actual[i]) <= tol, "$what[$i]: ${expected[i]} vs ${actual[i]}")
    }

    /** One DynamicTest per backend; the native one is skipped if it failed to load. */
    private fun perBackend(name: String, body: (LinAlgBackend) -> Unit): List<DynamicTest> = listOf(
        dynamicTest("$name [java]") { body(Backends.java()) },
        dynamicTest("$name [native]") {
            assumeTrue(Backends.isNativeAvailable(), "native BLAS/LAPACK is unavailable")
            body(Backends.native())
        },
    )

    private fun <T> cases(items: List<Pair<String, T>>, body: (LinAlgBackend, T) -> Unit): List<DynamicTest> =
        items.flatMap { (name, item) -> perBackend(name) { b -> body(b, item) } }

    // --- Backend registry -----------------------------------------------------------

    @TestFactory
    fun registry(): List<DynamicTest> = listOf(
        dynamicTest("available() is non-empty and contains java()") {
            val all = Backends.available()
            assertTrue(all.isNotEmpty())
            assertTrue(all.any { it === Backends.java() }, "java() must be in available()")
        },
        dynamicTest("available() contains native() exactly when isNativeAvailable()") {
            val hasNative = Backends.available().any { it.isNative }
            assertEquals(Backends.isNativeAvailable(), hasNative)
            if (hasNative) assertTrue(Backends.available().any { it === Backends.native() })
        },
        dynamicTest("backend names in available() are distinct") {
            val names = Backends.available().map { it.name }
            assertEquals(names.size, names.toSet().size, "duplicate names: $names")
        },
        dynamicTest("describe() contains the default backend name") {
            assertTrue(Backends.describe().contains(Backends.default().name), Backends.describe())
        },
        dynamicTest("resolve(java) is not native") { assertTrue(!Backends.resolve("java").isNative) },
        dynamicTest("resolve(native) is native") {
            assumeTrue(Backends.isNativeAvailable())
            assertTrue(Backends.resolve("native").isNative)
        },
        dynamicTest("resolve(auto) is native iff available") {
            assertEquals(Backends.isNativeAvailable(), Backends.resolve("auto").isNative)
        },
        dynamicTest("resolve(unknown) throws IllegalArgumentException listing the allowed values") {
            val e = assertFailsWith<IllegalArgumentException> { Backends.resolve("unknown") }
            val msg = e.message!!
            assertTrue(msg.contains("native") && msg.contains("java") && msg.contains("auto"), msg)
        },
    )

    // --- 1×1 matrices -------------------------------------------------------------------

    @TestFactory
    fun oneByOne(): List<DynamicTest> = perBackend("1×1") { b ->
        assertVec(v(2.0), LinearAlgebra.solve(arrayOf(v(2.0)), v(4.0), b), what = "solve")
        assertVec(v(0.25), Conditioning.inverse(m(v(4.0)), b)!!.data, what = "inverse")
        assertVec(v(3.0), LinearAlgebra.cholesky(m(v(9.0)), b)!!.data, what = "cholesky")
        assertVec(v(7.0), Conditioning.symmetricEigenvalues(m(v(7.0)), b), what = "eig")
        assertEquals(1.0, Conditioning.conditionInf(m(v(1.0)), backend = b).condInf, 1e-15, "condInf")
        assertVec(v(6.0), LinearAlgebra.matMat(m(v(2.0)), m(v(3.0)), b).data, what = "matMat")
    }

    // --- Rectangular -------------------------------------------------------------------

    @TestFactory
    fun rectangular(): List<DynamicTest> = perBackend("rectangular") { b ->
        val a35 = DenseMatrix.build(3, 5) { i, j -> (i + 2 * j).toDouble() }
        assertEquals(3, LinearAlgebra.matVec(a35, DoubleArray(5) { 1.0 }, b).size, "matVec 3×5·5")
        assertEquals(5, LinearAlgebra.matTransVec(a35, DoubleArray(3) { 1.0 }, b).size, "matTransVec 3×5ᵀ·3")
        val a23 = DenseMatrix.build(2, 3) { i, j -> (i + j).toDouble() }
        val a34 = DenseMatrix.build(3, 4) { i, j -> (i - j).toDouble() }
        val p = LinearAlgebra.matMat(a23, a34, b)
        assertEquals(2, p.rows); assertEquals(4, p.cols)
        assertFailsWith<IllegalArgumentException>("matMat 2×3·2×3") { LinearAlgebra.matMat(a23, a23, b) }
        assertFailsWith<IllegalArgumentException>("solve 3×4") { LinearAlgebra.solve(a34, v(1.0, 2.0, 3.0), b) }
        assertFailsWith<IllegalArgumentException>("cholesky 3×4") { LinearAlgebra.cholesky(a34, b) }
        for (k in MatrixNorm.values()) {
            val n = b.norm(a35, k)
            assertTrue(n.isFinite() && n >= 0.0, "norm $k = $n")
        }
    }

    // --- Empty ---------------------------------------------------------------------------

    @TestFactory
    fun empty(): List<DynamicTest> = perBackend("empty") { b ->
        assertFailsWith<IllegalArgumentException>("matVec 0×0") {
            LinearAlgebra.matVec(DenseMatrix.zeros(0, 0), DoubleArray(0), b)
        }
        assertEquals(0, DenseMatrix.zeros(0, 5).toRows().size)
        assertEquals(0, DenseMatrix.fromRows(emptyArray()).rows)
    }

    // --- Large sizes ---------------------------------------------------------------------

    @TestFactory
    fun large(): List<DynamicTest> {
        val native = Backends.isNativeAvailable()
        val b = if (native) Backends.native() else Backends.java()
        val nSolve = if (native) 1000 else 300
        val nEig = if (native) 500 else 200
        val tag = if (native) "native" else "java"
        return listOf(
            dynamicTest("solve on a diagonally dominant $nSolve×$nSolve [$tag]") {
                val a = DenseMatrix.fromRows(GoldenInputs.dd(nSolve))
                val rhs = GoldenInputs.vec(nSolve)
                val x = LinearAlgebra.solve(a, rhs, b)
                val r = LinearAlgebra.matVec(a, x, b)
                var res = 0.0
                for (i in rhs.indices) res = maxOf(res, abs(r[i] - rhs[i]))
                val scale = b.norm(a, MatrixNorm.INF) * LinearAlgebra.normInf(x) + LinearAlgebra.normInf(rhs)
                assertTrue(res <= 1e-10 * scale, "residual $res at scale $scale")
            },
            dynamicTest("symmetricEigenvalues on a symmetric $nEig×$nEig: sum = trace [$tag]") {
                val s = DenseMatrix.fromRows(GoldenInputs.sym(nEig))
                val eig = Conditioning.symmetricEigenvalues(s, b)
                var trace = 0.0
                for (i in 0 until nEig) trace += s[i, i]
                assertEquals(nEig, eig.size)
                assertTrue(abs(eig.sum() - trace) <= 1e-8 * b.norm(s, MatrixNorm.INF), "Σλ=${eig.sum()} vs tr=$trace")
            },
            dynamicTest("conditionEstimate on a diagonally dominant $nSolve×$nSolve in reasonable time [$tag]") {
                val a = DenseMatrix.fromRows(GoldenInputs.dd(nSolve))
                val est = assertTimeoutPreemptively(Duration.ofSeconds(10), ThrowingSupplier { Conditioning.conditionEstimate(a, backend = b) })
                assertTrue(est.condInf.isFinite() && est.condInf >= 1.0, "cond=${est.condInf}")
            },
        )
    }

    // --- Singular matrices of different structure -----------------------------------------------------------

    @TestFactory
    fun singularStructures(): List<DynamicTest> = perBackend("singular") { b ->
        val zero = DenseMatrix.zeros(3, 3)
        assertFailsWith<IllegalStateException>("solve on zero matrix") { LinearAlgebra.solve(zero, v(1.0, 2.0, 3.0), b) }
        assertNull(Conditioning.inverse(zero, b), "inverse of zero matrix")
        assertEquals(Double.POSITIVE_INFINITY, Conditioning.conditionInf(zero, backend = b).condInf, "conditionInf of zero matrix")
        assertEquals(Double.POSITIVE_INFINITY, Conditioning.conditionEstimate(zero, backend = b).condInf, "conditionEstimate of zero matrix")
        val rankDeficient = arrayOf(v(1.0, 2.0, 3.0), v(2.0, 4.0, 6.0), v(1.0, 1.0, 1.0))
        assertFailsWith<IllegalStateException>("rank deficient") { LinearAlgebra.solve(rankDeficient, v(1.0, 2.0, 3.0), b) }
        // A zero diagonal entry in a non-singular matrix is a permutation, not singularity.
        assertVec(v(2.0, 1.0), LinearAlgebra.solve(arrayOf(v(0.0, 1.0), v(1.0, 0.0)), v(1.0, 2.0), b), what = "permutation")
    }

    // --- Numerically extreme --------------------------------------------------------------------

    @TestFactory
    fun extremeScales(): List<DynamicTest> = cases(
        listOf("1e300·I" to 1e300, "1e-300·I" to 1e-300),
    ) { b, scale ->
        val a = arrayOf(v(scale, 0.0, 0.0), v(0.0, scale, 0.0), v(0.0, 0.0, scale))
        val rhs = v(1.0, 2.0, 3.0)
        val x = LinearAlgebra.solve(a, DoubleArray(3) { rhs[it] * scale }, b)
        assertVec(rhs, x, tol = 1e-15, what = "scale=$scale")
    } + perBackend("Double.MIN_VALUE on the diagonal") { b ->
        val a = arrayOf(v(Double.MIN_VALUE, 0.0), v(0.0, Double.MIN_VALUE))
        try {
            val x = LinearAlgebra.solve(a, v(1.0, 1.0), b)
            assertTrue(x.all { it.isFinite() }, "without an exception the solution must be finite: ${x.toList()}")
        } catch (_: IllegalStateException) {
            // Acceptable outcome: the backend honestly refused.
        }
    }

    // --- Quadrature ----------------------------------------------------------------------------

    @TestFactory
    fun quadrature(): List<DynamicTest> = listOf(
        dynamicTest("m=64: Σw=2, nodes in (−1, 1)") {
            val (x, w) = GaussLegendre.gaussLegendreReference(64)
            assertEquals(2.0, w.sum(), 1e-14)
            assertTrue(x.all { it > -1.0 && it < 1.0 })
            val (x2, w2) = GaussLegendre(64).refNodesWeights()
            assertVec(x, x2, 0.0, "refNodes"); assertVec(w, w2, 0.0, "refWeights")
        },
        dynamicTest("m=200: Σw=2 within 1e-13, nodes in (−1, 1)") {
            val (x, w) = GaussLegendre.gaussLegendreReference(200)
            assertEquals(2.0, w.sum(), 1e-13)
            assertTrue(x.all { it > -1.0 && it < 1.0 })
            assertEquals(200, x.toSet().size, "nodes are distinct")
        },
        dynamicTest("interval of length 1e-12 gives a finite result") {
            val r = GaussLegendre(8).integrate(v(0.0, 1e-12)) { t -> t * t + 1.0 }
            assertTrue(r.isFinite() && abs(r - 1e-12) <= 1e-24, "r=$r")
        },
        dynamicTest("integral of 1 over [−1e6, 1e6] is 2e6") {
            assertEquals(2e6, GaussLegendre(8).integrate(v(-1e6, 1e6)) { 1.0 }, 2e6 * 1e-9)
        },
    )

    // --- Parallel assembly ---------------------------------------------------------------------

    @TestFactory
    fun parallelAssembly(): List<DynamicTest> = listOf(
        dynamicTest("assembleDense 1×100000 matches sequential") {
            val f = { i: Int, j: Int -> i * 3.0 + j * 0.5 }
            val par = ParallelAssembly.assembleDense(1, 100_000, cellFn = f)
            val seq = ParallelAssembly.assembleDense(1, 100_000, parallel = false, cellFn = f)
            assertTrue(par.data.contentEquals(seq.data))
        },
        dynamicTest("assembleDense 100000×1 matches sequential") {
            val f = { i: Int, j: Int -> i * 3.0 + j * 0.5 }
            val par = ParallelAssembly.assembleDense(100_000, 1, cellFn = f)
            val seq = ParallelAssembly.assembleDense(100_000, 1, parallel = false, cellFn = f)
            assertTrue(par.data.contentEquals(seq.data))
        },
        dynamicTest("nested assembleDense on the common pool does not hang") {
            val nested = { i: Int, j: Int ->
                ParallelAssembly.assembleDense(3, 3) { a, c -> (a + c).toDouble() }.data.sum() + i + j
            }
            val par = assertTimeoutPreemptively(Duration.ofSeconds(30), ThrowingSupplier { ParallelAssembly.assembleDense(4, 4, cellFn = nested) })
            val seq = ParallelAssembly.assembleDense(4, 4, parallel = false, cellFn = nested)
            assertTrue(par.data.contentEquals(seq.data))
        },
    )

    // --- measured ----------------------------------------------------------------------------------

    @TestFactory
    fun measuredBoundaries(): List<DynamicTest> = listOf(
        dynamicTest("measured(-0.0) is AtNoiseLevel, like 0.0") { assertIs<Measured.AtNoiseLevel>(measured(-0.0)) },
        dynamicTest("measured(Double.MIN_VALUE) is AtNoiseLevel") { assertIs<Measured.AtNoiseLevel>(measured(Double.MIN_VALUE)) },
        dynamicTest("measured(1e-13, 1e-13): boundary included, |value| >= threshold counts as reliable") {
            assertIs<Measured.Reliable>(measured(1e-13, 1e-13))
            assertIs<Measured.AtNoiseLevel>(measured(Math.nextDown(1e-13), 1e-13))
        },
        dynamicTest("measured(NaN) is AtNoiseLevel, value preserved") {
            val r = measured(Double.NaN)
            assertNotNull(r)
            assertIs<Measured.AtNoiseLevel>(r)
            assertTrue(r.value.isNaN())
        },
    )
}
