package numerics.openblas

import numerics.Conditioning
import numerics.DenseMatrix
import numerics.LinearAlgebra
import numerics.backend.Backends
import org.junit.jupiter.api.BeforeAll
import java.io.File
import java.util.Random
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class OpenBlasTest {

    companion object {
        @JvmStatic
        @BeforeAll
        fun setUp() {
            // Before any access to Backends: the path is read once at initialization.
            assertTrue(OpenBlas.install(), "OpenBLAS must unpack on this platform")
        }

        fun diagonallyDominant(n: Int): Array<DoubleArray> {
            val r = Random(1000L + n)
            return Array(n) { i ->
                DoubleArray(n) { r.nextDouble() * 2 - 1 }.also { row -> row[i] += n.toDouble() }
            }
        }

        fun rhs(n: Int): DoubleArray {
            val r = Random(2000L + n)
            return DoubleArray(n) { r.nextDouble() * 2 - 1 }
        }

        fun symmetric(n: Int): Array<DoubleArray> {
            val r = Random(3000L + n)
            val a = Array(n) { DoubleArray(n) }
            for (i in 0 until n) {
                for (j in i until n) {
                    val v = r.nextDouble() * 2 - 1
                    a[i][j] = v
                    a[j][i] = v
                }
            }
            return a
        }

        fun maxAbsDiff(x: DoubleArray, y: DoubleArray): Double {
            assertEquals(x.size, y.size)
            var m = 0.0
            for (i in x.indices) m = max(m, abs(x[i] - y[i]))
            return m
        }

        fun maxAbs(x: DoubleArray): Double = x.fold(0.0) { m, v -> max(m, abs(v)) }
    }

    @Test
    fun `library path exists and points to the LAPACK variant`() {
        val path = assertNotNull(OpenBlas.libraryPath())
        val file = File(path)
        assertTrue(file.exists(), "file $path must exist")
        val name = file.name
        assertTrue(name.contains("openblas"), name)
        assertFalse(name.contains("nolapack"), name)
        assertFalse(name.contains("jni"), name)
    }

    @Test
    fun `after install the library selects the native implementation`() {
        assertEquals(OpenBlas.libraryPath(), System.getProperty("dev.ludovic.netlib.lapack.nativeLibPath"))
        assertEquals(OpenBlas.libraryPath(), System.getProperty("dev.ludovic.netlib.blas.nativeLibPath"))
        assertTrue(OpenBlas.isInstalled())
        assertTrue(Backends.isNativeAvailable(), Backends.describe())
        assertTrue(Backends.default().isNative, Backends.describe())
        assertTrue(Backends.default().name.contains("JNI"), Backends.default().name)
    }

    @Test
    fun `solve on 1024 matches the JVM implementation and gives a small residual`() {
        val n = 1024
        val rows = diagonallyDominant(n)
        val a = DenseMatrix.fromRows(rows)
        val b = rhs(n)

        val start = System.nanoTime()
        val x = LinearAlgebra.solve(a, b, Backends.default())
        val elapsedMs = (System.nanoTime() - start) / 1e6
        println("OpenBLAS: solve n=$n in %.1f ms (%s)".format(elapsedMs, Backends.default().name))

        val residual = DoubleArray(n) { i ->
            var s = 0.0
            for (j in 0 until n) s += rows[i][j] * x[j]
            s - b[i]
        }
        val normA = Conditioning.matrixNormInf(a, Backends.default())
        val bound = 1e-10 * (normA * maxAbs(x) + maxAbs(b))
        assertTrue(maxAbs(residual) <= bound, "residual ${maxAbs(residual)} > $bound")

        val xJava = LinearAlgebra.solve(a, b, Backends.java())
        assertTrue(maxAbsDiff(x, xJava) <= 1e-10 * maxAbs(xJava), "discrepancy with the JVM implementation ${maxAbsDiff(x, xJava)}")
    }

    @Test
    fun `symmetricEigenvalues on 256 matches the JVM implementation`() {
        val a = DenseMatrix.fromRows(symmetric(256))
        val native = Conditioning.symmetricEigenvalues(a, Backends.default())
        val java = Conditioning.symmetricEigenvalues(a, Backends.java())
        assertTrue(maxAbsDiff(native, java) <= 1e-10 * max(1.0, maxAbs(java)), "discrepancy ${maxAbsDiff(native, java)}")
    }

    @Test
    fun `cholesky on 256 matches the JVM implementation`() {
        val n = 256
        val b = DenseMatrix.fromRows(symmetric(n))
        val ones = DoubleArray(n) { 1.0 }
        val spdRows = LinearAlgebra.atWa(b, ones, Backends.java())
        val rows = Array(n) { i -> DoubleArray(n) { j -> spdRows.data[i * n + j] } }
        for (i in 0 until n) rows[i][i] += n.toDouble()
        val spd = DenseMatrix.fromRows(rows)

        val native = assertNotNull(LinearAlgebra.cholesky(spd, Backends.default()))
        val java = assertNotNull(LinearAlgebra.cholesky(spd, Backends.java()))
        assertTrue(maxAbsDiff(native.data, java.data) <= 1e-10 * max(1.0, maxAbs(java.data)), "discrepancy ${maxAbsDiff(native.data, java.data)}")
    }

    @Test
    fun `repeated install is idempotent`() {
        val lapack = System.getProperty("dev.ludovic.netlib.lapack.nativeLibPath")
        val blas = System.getProperty("dev.ludovic.netlib.blas.nativeLibPath")
        assertTrue(OpenBlas.install())
        assertTrue(OpenBlas.install(threads = 1))
        assertEquals(lapack, System.getProperty("dev.ludovic.netlib.lapack.nativeLibPath"))
        assertEquals(blas, System.getProperty("dev.ludovic.netlib.blas.nativeLibPath"))
    }
}
