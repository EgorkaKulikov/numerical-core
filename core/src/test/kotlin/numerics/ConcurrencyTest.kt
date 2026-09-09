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
 * Потокобезопасность: реализации BLAS/LAPACK, реестр реализаций, квадратура и контекст
 * не имеют изменяемого общего состояния — параллельные вызовы дают побитово те же ответы,
 * что и последовательные, а ленивые синглтоны создаются ровно один раз.
 */
@Tag("fast")
class ConcurrencyTest {

    private val timeout: Duration = Duration.ofSeconds(60)

    /** Запускает [tasks] на пуле из [threads] потоков со стартом по общему сигналу; возвращает результаты. */
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

    private fun assertBitwise(expected: DoubleArray, actual: DoubleArray, what: String) {
        assertTrue(expected.contentEquals(actual), "$what: результат из потока отличается побитово")
    }

    private fun perBackend(name: String, body: (LinAlgBackend) -> Unit): List<DynamicTest> =
        Backends.available().map { b -> dynamicTest("$name [${b.name}]") { body(b) } }

    @TestFactory
    fun solveFromEightThreadsIsBitwiseReproducible(): List<DynamicTest> = perBackend("solve 8×20") { b ->
        assertTimeoutPreemptively(timeout) {
            val a = GoldenInputs.dd(64)
            val rhs = GoldenInputs.vec(64)
            val expected = LinearAlgebra.solve(a, rhs, b)
            val results = runAll(8, List(8) { { List(20) { LinearAlgebra.solve(a, rhs, b) } } })
            results.forEachIndexed { t, list ->
                list.forEachIndexed { k, x -> assertBitwise(expected, x, "поток $t, итерация $k") }
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
            val eigTol = 1e-12 * eSeq.maxOf { abs(it) }
            for ((kind, r) in runAll(12, tasks)) {
                when (kind) {
                    "solve" -> assertBitwise(xSeq, r, kind)
                    "matMat" -> assertBitwise(pSeq, r, kind)
                    // Спектр нативной LAPACK (Accelerate) под нагрузкой воспроизводится не побитово,
                    // а с точностью округления — сравниваем с допуском 1e-12 относительно max|λ|.
                    else -> {
                        var maxDiff = 0.0
                        for (i in eSeq.indices) maxDiff = maxOf(maxDiff, abs(eSeq[i] - r[i]))
                        if (!eSeq.contentEquals(r)) println("eig [${b.name}]: не побитово, max|Δλ| = $maxDiff")
                        assertTrue(maxDiff <= eigTol, "eig: max|Δλ| = $maxDiff > $eigTol")
                    }
                }
            }
        }
    }

    @Test
    fun backendRegistryReturnsSingletonsUnderContention() {
        assertTimeoutPreemptively(timeout) {
            val javas = runAll(16, List(16) { { Backends.java() } })
            javas.forEach { assertSame(javas[0], it, "java() обязан кэшировать экземпляр") }
            val defaults = runAll(16, List(16) { { Backends.default() } })
            defaults.forEach { assertSame(defaults[0], it, "default() обязан кэшировать экземпляр") }
            if (Backends.isNativeAvailable()) {
                val natives = runAll(16, List(16) { { Backends.native() } })
                natives.forEach { assertSame(natives[0], it, "native() обязан кэшировать экземпляр") }
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
                    assertTrue(r.toRawBits() == expected.toRawBits(), "поток $t: $r vs $expected")
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
