package numerics.golden

import numerics.GaussLegendre
import numerics.golden.GoldenIo.assertClose
import numerics.golden.GoldenIo.entry
import numerics.golden.GoldenIo.obj
import numerics.golden.GoldenIo.section
import numerics.golden.GoldenIo.toDbl
import numerics.golden.GoldenIo.toVec
import numerics.golden.GoldenInputs.DEFAULT_TOL
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory

/** Эталон `quadrature.json`: узлы/веса `gaussLegendreReference(m)` и составная квадратура. */
@Tag("fast")
class GoldenQuadratureTest {

    @TestFactory
    fun cases(): List<DynamicTest> {
        val root = GoldenIo.readGolden("quadrature.json")
        val reference = section(root, "reference")
        val integrate = section(root, "integrate")
        val refTests = GoldenInputs.referenceOrders.map { m ->
            DynamicTest.dynamicTest("gaussLegendreReference m=$m") {
                val exp = obj(entry(reference, m.toString()))
                val (nodes, weights) = GaussLegendre.gaussLegendreReference(m)
                assertClose(toVec(exp["nodes"]), nodes, DEFAULT_TOL, "m=$m.nodes")
                assertClose(toVec(exp["weights"]), weights, DEFAULT_TOL, "m=$m.weights")
            }
        }
        val intTests = GoldenInputs.integrateCases.map { case ->
            DynamicTest.dynamicTest("integrate ${case.key}") {
                val got = GaussLegendre(case.m).integrate(case.bp, case.f)
                assertClose(toDbl(entry(integrate, case.key)), got, DEFAULT_TOL, "${case.key}.integrate")
            }
        }
        return refTests + intTests
    }
}
