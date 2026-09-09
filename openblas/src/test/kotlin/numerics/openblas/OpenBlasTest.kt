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
            // До любого обращения к Backends: путь читается один раз при инициализации.
            assertTrue(OpenBlas.install(), "OpenBLAS должна распаковаться на этой платформе")
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
    fun `путь к библиотеке существует и указывает на вариант с LAPACK`() {
        val path = assertNotNull(OpenBlas.libraryPath())
        val file = File(path)
        assertTrue(file.exists(), "файл $path должен существовать")
        val name = file.name
        assertTrue(name.contains("openblas"), name)
        assertFalse(name.contains("nolapack"), name)
        assertFalse(name.contains("jni"), name)
    }

    @Test
    fun `после установки библиотека выбирает нативную реализацию`() {
        assertEquals(OpenBlas.libraryPath(), System.getProperty("dev.ludovic.netlib.lapack.nativeLibPath"))
        assertEquals(OpenBlas.libraryPath(), System.getProperty("dev.ludovic.netlib.blas.nativeLibPath"))
        assertTrue(OpenBlas.isInstalled())
        assertTrue(Backends.isNativeAvailable(), Backends.describe())
        assertTrue(Backends.default().isNative, Backends.describe())
        assertTrue(Backends.default().name.contains("JNI"), Backends.default().name)
    }

    @Test
    fun `solve на 1024 совпадает с JVM-реализацией и даёт малую невязку`() {
        val n = 1024
        val rows = diagonallyDominant(n)
        val a = DenseMatrix.fromRows(rows)
        val b = rhs(n)

        val start = System.nanoTime()
        val x = LinearAlgebra.solve(a, b, Backends.default())
        val elapsedMs = (System.nanoTime() - start) / 1e6
        println("OpenBLAS: solve n=$n за %.1f мс (%s)".format(elapsedMs, Backends.default().name))

        val residual = DoubleArray(n) { i ->
            var s = 0.0
            for (j in 0 until n) s += rows[i][j] * x[j]
            s - b[i]
        }
        val normA = Conditioning.matrixNormInf(a, Backends.default())
        val bound = 1e-10 * (normA * maxAbs(x) + maxAbs(b))
        assertTrue(maxAbs(residual) <= bound, "невязка ${maxAbs(residual)} > $bound")

        val xJava = LinearAlgebra.solve(a, b, Backends.java())
        assertTrue(maxAbsDiff(x, xJava) <= 1e-10 * maxAbs(xJava), "расхождение с JVM-реализацией ${maxAbsDiff(x, xJava)}")
    }

    @Test
    fun `symmetricEigenvalues на 256 совпадает с JVM-реализацией`() {
        val a = DenseMatrix.fromRows(symmetric(256))
        val native = Conditioning.symmetricEigenvalues(a, Backends.default())
        val java = Conditioning.symmetricEigenvalues(a, Backends.java())
        assertTrue(maxAbsDiff(native, java) <= 1e-10 * max(1.0, maxAbs(java)), "расхождение ${maxAbsDiff(native, java)}")
    }

    @Test
    fun `cholesky на 256 совпадает с JVM-реализацией`() {
        val n = 256
        val b = DenseMatrix.fromRows(symmetric(n))
        val ones = DoubleArray(n) { 1.0 }
        val spdRows = LinearAlgebra.atWa(b, ones, Backends.java())
        val rows = Array(n) { i -> DoubleArray(n) { j -> spdRows.data[i * n + j] } }
        for (i in 0 until n) rows[i][i] += n.toDouble()
        val spd = DenseMatrix.fromRows(rows)

        val native = assertNotNull(LinearAlgebra.cholesky(spd, Backends.default()))
        val java = assertNotNull(LinearAlgebra.cholesky(spd, Backends.java()))
        assertTrue(maxAbsDiff(native.data, java.data) <= 1e-10 * max(1.0, maxAbs(java.data)), "расхождение ${maxAbsDiff(native.data, java.data)}")
    }

    @Test
    fun `повторная установка идемпотентна`() {
        val lapack = System.getProperty("dev.ludovic.netlib.lapack.nativeLibPath")
        val blas = System.getProperty("dev.ludovic.netlib.blas.nativeLibPath")
        assertTrue(OpenBlas.install())
        assertTrue(OpenBlas.install(threads = 1))
        assertEquals(lapack, System.getProperty("dev.ludovic.netlib.lapack.nativeLibPath"))
        assertEquals(blas, System.getProperty("dev.ludovic.netlib.blas.nativeLibPath"))
    }
}
