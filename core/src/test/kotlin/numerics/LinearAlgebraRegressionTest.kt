package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Regression tests for the linear algebra layer: singularity is detected regardless of the
 * data scale, and non-finite input (NaN, infinity) is rejected with an exception rather than
 * silently returning a non-finite answer. Every case runs on all available backends.
 */
@Tag("fast")
class LinearAlgebraRegressionTest {

    private fun onAll(name: String, body: (LinAlgBackend) -> Unit): List<DynamicTest> =
        Backends.available().map { b -> DynamicTest.dynamicTest("$name [${b.name}]") { body(b) } }

    /** One DynamicTest per (backend, case) pair. */
    private fun <T> onAllWith(name: String, cases: List<T>, body: (LinAlgBackend, T) -> Unit): List<DynamicTest> =
        Backends.available().flatMap { b ->
            cases.map { c -> DynamicTest.dynamicTest("$name $c [${b.name}]") { body(b, c) } }
        }

    @TestFactory
    fun singularityDetectedRegardlessOfScale() = onAllWith("scale", listOf(1.0, 1e6, 1e-6)) { backend, scale ->
        val singular = arrayOf(doubleArrayOf(1.0 * scale, 2.0 * scale), doubleArrayOf(2.0 * scale, 4.0 * scale))
        assertFailsWith<IllegalStateException>("Scale $scale") {
            LinearAlgebra.solve(singular, doubleArrayOf(1.0 * scale, 2.0 * scale), backend)
        }
        val regular = arrayOf(doubleArrayOf(1.0 * scale, 2.0 * scale), doubleArrayOf(3.0 * scale, 4.0 * scale))
        val x = LinearAlgebra.solve(regular, doubleArrayOf(1.0 * scale, 1.0 * scale), backend)
        assertTrue(x.all { it.isFinite() }, "Scale $scale: solution must be finite")
    }

    @TestFactory
    fun nonFiniteInputRejected() = onAllWith(
        "non-finite input",
        listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY),
    ) { backend, bad ->
        val a = arrayOf(doubleArrayOf(bad, 1.0), doubleArrayOf(1.0, 1.0))
        val ex = assertFailsWith<IllegalStateException>("A contains $bad") {
            LinearAlgebra.solve(a, doubleArrayOf(1.0, 1.0), backend)
        }
        assertTrue(ex.message!!.contains("non-finite"), ex.message)
        val good = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))
        assertFailsWith<IllegalStateException>("b contains $bad") {
            LinearAlgebra.solve(good, doubleArrayOf(bad, 1.0), backend)
        }
    }
}
