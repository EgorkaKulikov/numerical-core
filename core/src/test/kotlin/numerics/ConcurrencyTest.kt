package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.golden.GoldenInputs
import org.junit.jupiter.api.Assertions.assertTimeoutPreemptively
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.DynamicTest.dynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import java.time.Duration
import java.util.concurrent.Callable
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Thread safety: BLAS/LAPACK implementations, the implementation registry, quadrature and the context
 * have no mutable shared state — concurrent calls return the same answers as
 * sequential ones, and lazy singletons are created exactly once.
 *
 * For the Java implementation the results match bit for bit; native multithreaded libraries
 * (Accelerate, OpenBLAS, MKL) reproduce the result only up to the reduction order —
 * the discrepancy does not exceed a few ulps, so for them the results
 * are compared with the relative tolerance [nativeTol] (see `docs/ACCURACY.md`, section
 * "Reproducibility on multithreaded implementations").
 */
@Tag("fast")
class ConcurrencyTest {

    private val timeout: Duration = Duration.ofSeconds(60)

    /** Tolerance for native implementations, relative to `max(‖expected‖∞, 1)`. */
    private val nativeTol: Double = 1e-13

    /** Runs [tasks] on a pool of [threads] threads started by a shared signal; returns the results. */
    private fun <T> runAll(threads: Int, tasks: List<() -> T>): List<T> {
        val pool = Executors.newFixedThreadPool(threads)
        val start = CountDownLatch(1)
        try {
            val futures = tasks.map { task ->
                pool.submit(Callable {
                    start.await()
                    task()
                })
            }
            start.countDown()
            return futures.map {
                try {
                    it.get()
                } catch (e: ExecutionException) {
                    throw e.cause ?: e
                }
            }
        } finally {
            pool.shutdown()
            pool.awaitTermination(30, TimeUnit.SECONDS)
        }
    }

    /**
     * Compares the result from a worker thread with the main-thread result: bit for bit for
     * the Java implementation, with relative tolerance [nativeTol] for native implementations.
     */
    private fun assertSameResult(expected: DoubleArray, actual: DoubleArray, backend: LinAlgBackend, label: String) {
        val tag = "$label [${backend.name}]"
        assertTrue(expected.size == actual.size, "$tag: size ${actual.size} != ${expected.size}")
        if (!backend.isNative) {
            assertTrue(expected.contentEquals(actual), "$tag: result from the thread differs bitwise")
            return
        }
        var maxDiff = 0.0
        var scale = 1.0
        for (i in expected.indices) {
            maxDiff = maxOf(maxDiff, abs(expected[i] - actual[i]))
            scale = maxOf(scale, abs(expected[i]))
        }
        val rel = maxDiff / scale
        if (maxDiff != 0.0) println("$tag: not bitwise equal, max|Δ| = $maxDiff (rel. $rel)")
        assertTrue(rel <= nativeTol, "$tag: relative discrepancy $rel > $nativeTol")
    }

    private fun perBackend(name: String, body: (LinAlgBackend) -> Unit): List<DynamicTest> =
        Backends.available().map { b -> dynamicTest("$name [${b.name}]") { body(b) } }

    @TestFactory
    fun solveFromEightThreadsIsReproducible(): List<DynamicTest> = perBackend("solve 8×20") { b ->
        assertTimeoutPreemptively(timeout) {
            val a = GoldenInputs.dd(64)
            val rhs = GoldenInputs.vec(64)
            val expected = LinearAlgebra.solve(a, rhs, b)
            val results = runAll(8, List(8) { { List(20) { LinearAlgebra.solve(a, rhs, b) } } })
            results.forEachIndexed { t, list ->
                list.forEachIndexed { k, x -> assertSameResult(expected, x, b, "thread $t, iteration $k") }
            }
        }
    }

    @TestFactory
    fun mixedOperationsInParallel(): List<DynamicTest> = perBackend("solve+matMat+eig ×4") { b ->
        assertTimeoutPreemptively(timeout) {
            val a = DenseMatrix.fromRows(GoldenInputs.dd(64))
            val s = DenseMatrix.fromRows(GoldenInputs.sym(64))
            val rhs = GoldenInputs.vec(64)
            val xSeq = LinearAlgebra.solve(a, rhs, b)
            val pSeq = LinearAlgebra.matMat(a, s, b).data
            val eSeq = Conditioning.symmetricEigenvalues(s, b)
            val tasks: List<() -> Pair<String, DoubleArray>> =
                List(4) { { "solve" to LinearAlgebra.solve(a, rhs, b) } } +
                    List(4) { { "matMat" to LinearAlgebra.matMat(a, s, b).data } } +
                    List(4) { { "eig" to Conditioning.symmetricEigenvalues(s, b) } }
            for ((kind, r) in runAll(12, tasks)) {
                when (kind) {
                    "solve" -> assertSameResult(xSeq, r, b, kind)
                    "matMat" -> assertSameResult(pSeq, r, b, kind)
                    else -> assertSameResult(eSeq, r, b, kind)
                }
            }
        }
    }

    @Test
    fun backendRegistryReturnsSingletonsUnderContention() {
        assertTimeoutPreemptively(timeout) {
            val javas = runAll(16, List(16) { { Backends.java() } })
            javas.forEach { assertSame(javas[0], it, "java() must cache the instance") }
            val defaults = runAll(16, List(16) { { Backends.default() } })
            defaults.forEach { assertSame(defaults[0], it, "default() must cache the instance") }
            if (Backends.isNativeAvailable()) {
                val natives = runAll(16, List(16) { { Backends.native() } })
                natives.forEach { assertSame(natives[0], it, "native() must cache the instance") }
            }
        }
    }

    @Test
    fun sharedGaussLegendreIsImmutableUnderConcurrentIntegrate() {
        assertTimeoutPreemptively(timeout) {
            val q = GaussLegendre(8)
            val bp = doubleArrayOf(0.0, 0.3, 1.0)
            val f = { t: Double -> Math.exp(-t * t) * (1.0 + t) }
            val expected = q.integrate(bp, f)
            val results = runAll(8, List(8) { { DoubleArray(50) { q.integrate(bp, f) } } })
            results.forEachIndexed { t, arr ->
                arr.forEach { r ->
                    assertTrue(r.toRawBits() == expected.toRawBits(), "thread $t: $r vs $expected")
                }
            }
        }
    }

    @Test
    fun numericsContextDefaultIsSingleton() {
        assertTimeoutPreemptively(timeout) {
            val ctxs = runAll(8, List(8) { { NumericsContext.default() } })
            ctxs.forEach { assertSame(ctxs[0], it) }
        }
    }
}
