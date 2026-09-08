package numerics.backend

import numerics.DenseMatrix
import numerics.LinearAlgebra
import numerics.NumericsContext
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.Tag
import kotlin.math.abs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Контракт [LinAlgBackend]/[Backends]: доступность реализаций, выбор по свойству
 * `numerics.backend`, согласие нативной и Java-реализаций, делегирование фасада
 * явно переданной реализации и единая семантика вырожденности.
 */
@Tag("fast")
class BackendSpiTest {

    private val tol = 1e-10

    private fun rand(rnd: Random, rows: Int, cols: Int) = DenseMatrix.build(rows, cols) { _, _ -> rnd.nextDouble(-1.0, 1.0) }

    private fun diagDominant(rnd: Random, n: Int): DenseMatrix {
        val a = rand(rnd, n, n)
        for (i in 0 until n) {
            var s = 0.0
            for (j in 0 until n) s += abs(a[i, j])
            a[i, i] += s + 1.0
        }
        return a
    }

    private fun assertClose(expected: DoubleArray, actual: DoubleArray, what: String) {
        assertEquals(expected.size, actual.size, what)
        for (i in expected.indices) assertTrue(abs(expected[i] - actual[i]) < tol, "$what[$i]: ${expected[i]} vs ${actual[i]}")
    }

    @Test
    fun javaBackendAlwaysAvailable() {
        val j = Backends.java()
        assertFalse(j.isNative)
        assertTrue(Backends.available().contains(j))
    }

    @Test
    fun nativeBackendAvailableHere() {
        assumeTrue(Backends.isNativeAvailable(), "нативная BLAS/LAPACK на этой машине не загрузилась")
        assertTrue(Backends.native().isNative)
        assertEquals(2, Backends.available().size)
        assertTrue(Backends.available()[0].isNative)
    }

    @Test
    fun defaultMatchesRequestedProperty() {
        when (System.getProperty("numerics.backend")?.trim()?.lowercase()) {
            "java" -> assertFalse(Backends.default().isNative)
            "native" -> assertTrue(Backends.default().isNative)
            else -> assertEquals(Backends.isNativeAvailable(), Backends.default().isNative)
        }
        assertEquals(Backends.default(), NumericsContext.default().backend)
        assertTrue(Backends.describe().contains(Backends.default().name))
    }

    @Test
    fun resolveHonoursModes() {
        assertFalse(Backends.resolve("java").isNative)
        assertFalse(Backends.resolve(" Java ").isNative)
        if (Backends.isNativeAvailable()) {
            assertTrue(Backends.resolve("native").isNative)
            assertTrue(Backends.resolve("auto").isNative)
            assertTrue(Backends.resolve(null).isNative)
        } else {
            assertFailsWith<IllegalStateException> { Backends.resolve("native") }
            assertFalse(Backends.resolve(null).isNative)
        }
        val ex = assertFailsWith<IllegalArgumentException> { Backends.resolve("cuda") }
        assertTrue(ex.message!!.contains("native, java, auto"), ex.message)
    }

    @Test
    fun nativeAndJavaAgreeOnRandomSystems() {
        assumeTrue(Backends.isNativeAvailable())
        val nat = Backends.native()
        val jav = Backends.java()
        for (n in intArrayOf(3, 8, 20, 50)) {
            val rnd = Random(7000 + n)
            val a = diagDominant(rnd, n)
            val b = rand(rnd, n, 2)
            val x = DoubleArray(n) { rnd.nextDouble(-1.0, 1.0) }
            assertClose(jav.solve(a, b).data, nat.solve(a, b).data, "solve n=$n")
            assertClose(jav.matVec(a, x), nat.matVec(a, x), "matVec n=$n")
            assertClose(jav.matTransVec(a, x), nat.matTransVec(a, x), "matTransVec n=$n")
            assertClose(jav.matMat(a, b).data, nat.matMat(a, b).data, "matMat n=$n")
            assertClose(jav.matTransMat(a, b).data, nat.matTransMat(a, b).data, "matTransMat n=$n")
            assertClose(jav.inverse(a)!!.data, nat.inverse(a)!!.data, "inverse n=$n")
            val lu = jav.luFactor(a)
            assertClose(jav.luSolve(lu, b).data, nat.luSolve(nat.luFactor(a), b).data, "luSolve n=$n")
            val spd = jav.matTransMat(a, a)
            assertClose(jav.cholesky(spd)!!.data, nat.cholesky(spd)!!.data, "cholesky n=$n")
            assertClose(jav.symmetricEigenvalues(spd), nat.symmetricEigenvalues(spd), "eig n=$n")
            for (k in MatrixNorm.values()) assertClose(doubleArrayOf(jav.norm(a, k)), doubleArrayOf(nat.norm(a, k)), "norm $k")
            val r1 = jav.reciprocalCondition1(lu, jav.norm(a, MatrixNorm.ONE))
            val r2 = nat.reciprocalCondition1(nat.luFactor(a), nat.norm(a, MatrixNorm.ONE))
            assertTrue(abs(r1 - r2) < 1e-6, "rcond $r1 vs $r2")
        }
    }

    @Test
    fun facadeUsesExplicitlyPassedBackend() {
        val a = arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 4.0))
        val expected = doubleArrayOf(1.0, 2.0)
        for (backend in Backends.available()) {
            assertClose(expected, LinearAlgebra.solve(a, doubleArrayOf(2.0, 8.0), backend), backend.name)
            assertClose(expected, LinearAlgebra.matVec(LinearAlgebra.identity(2), expected, backend), backend.name)
        }
        assertTrue(NumericsContext.default().parallel)
    }

    @Test
    fun singularSystemThrowsOnAllBackends() {
        val a = DenseMatrix.fromRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        val b = DenseMatrix.fromColumnMajor(2, 1, doubleArrayOf(1.0, 2.0))
        for (backend in Backends.available()) {
            assertFailsWith<IllegalStateException>(backend.name) { backend.solve(a, b) }
            assertTrue(backend.luFactor(a).isSingular, backend.name)
            assertEquals(null, backend.inverse(a), backend.name)
            assertEquals(null, backend.cholesky(DenseMatrix.fromRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 1.0)))))
            assertFailsWith<IllegalStateException>(backend.name) { LinearAlgebra.solve(a, doubleArrayOf(1.0, 2.0), backend) }
        }
    }
}
