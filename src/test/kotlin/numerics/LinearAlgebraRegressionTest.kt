package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Регрессионные тесты слоя линейной алгебры: вырожденность распознаётся независимо от
 * масштаба данных, а нечисловой вход (NaN, бесконечность) отвергается исключением, а не
 * молчаливым нечисловым ответом. Каждый случай гоняется на всех доступных реализациях.
 */
@Tag("fast")
class LinearAlgebraRegressionTest {

    private fun onAll(name: String, body: (LinAlgBackend) -> Unit): List<DynamicTest> =
        Backends.available().map { b -> DynamicTest.dynamicTest("$name [${b.name}]") { body(b) } }

    /** Один DynamicTest на пару (реализация, случай). */
    private fun <T> onAllWith(name: String, cases: List<T>, body: (LinAlgBackend, T) -> Unit): List<DynamicTest> =
        Backends.available().flatMap { b ->
            cases.map { c -> DynamicTest.dynamicTest("$name $c [${b.name}]") { body(b, c) } }
        }

    @TestFactory
    fun singularityDetectedRegardlessOfScale() = onAllWith("масштаб", listOf(1.0, 1e6, 1e-6)) { backend, scale ->
        val singular = arrayOf(doubleArrayOf(1.0 * scale, 2.0 * scale), doubleArrayOf(2.0 * scale, 4.0 * scale))
        assertFailsWith<IllegalStateException>("Масштаб $scale") {
            LinearAlgebra.solve(singular, doubleArrayOf(1.0 * scale, 2.0 * scale), backend)
        }
        val regular = arrayOf(doubleArrayOf(1.0 * scale, 2.0 * scale), doubleArrayOf(3.0 * scale, 4.0 * scale))
        val x = LinearAlgebra.solve(regular, doubleArrayOf(1.0 * scale, 1.0 * scale), backend)
        assertTrue(x.all { it.isFinite() }, "Масштаб $scale: решение должно быть конечным")
    }

    @TestFactory
    fun nonFiniteInputRejected() = onAllWith(
        "нечисловой вход",
        listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY),
    ) { backend, bad ->
        val a = arrayOf(doubleArrayOf(bad, 1.0), doubleArrayOf(1.0, 1.0))
        val ex = assertFailsWith<IllegalStateException>("A содержит $bad") {
            LinearAlgebra.solve(a, doubleArrayOf(1.0, 1.0), backend)
        }
        assertTrue(ex.message!!.contains("нечисловые"), ex.message)
        val good = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))
        assertFailsWith<IllegalStateException>("b содержит $bad") {
            LinearAlgebra.solve(good, doubleArrayOf(bad, 1.0), backend)
        }
    }
}
