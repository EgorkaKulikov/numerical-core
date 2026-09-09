package numerics.openblas

import numerics.DenseMatrix
import numerics.LinearAlgebra
import numerics.backend.Backends
import org.junit.jupiter.api.BeforeAll
import java.util.Random
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.abs
import kotlin.math.max
import kotlin.test.Test
import kotlin.test.assertNull
import kotlin.test.assertTrue

class OpenBlasThreadsTest {

    companion object {
        @JvmStatic
        @BeforeAll
        fun setUp() {
            assertTrue(OpenBlas.install(), "OpenBLAS должна распаковаться на этой платформе")
        }
    }

    @Test
    fun `solve на 1024 в потоке с явным стеком 8 МБ проходит без ошибок`() {
        val n = 1024
        val r = Random(1000L + n)
        val rows = Array(n) { i ->
            DoubleArray(n) { r.nextDouble() * 2 - 1 }.also { row -> row[i] += n.toDouble() }
        }
        val rb = Random(2000L + n)
        val b = DoubleArray(n) { rb.nextDouble() * 2 - 1 }
        val a = DenseMatrix.fromRows(rows)

        val failure = AtomicReference<Throwable>()
        val worker = Thread(null, {
            try {
                assertTrue(Backends.default().isNative, Backends.describe())
                val x = LinearAlgebra.solve(a, b, Backends.default())
                var residual = 0.0
                for (i in 0 until n) {
                    var s = 0.0
                    for (j in 0 until n) s += rows[i][j] * x[j]
                    residual = max(residual, abs(s - b[i]))
                }
                assertTrue(residual <= 1e-8, "невязка $residual")
            } catch (t: Throwable) {
                failure.set(t)
            }
        }, "lapack", 8L shl 20)
        worker.start()
        worker.join()
        assertNull(failure.get(), "ошибка в рабочем потоке: ${failure.get()}")
    }
}
