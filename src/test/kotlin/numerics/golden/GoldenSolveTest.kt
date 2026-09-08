package numerics.golden

import numerics.Conditioning
import numerics.ForwardError
import numerics.LinearAlgebra
import numerics.golden.GoldenIo.assertClose
import numerics.golden.GoldenIo.entry
import numerics.golden.GoldenIo.obj
import numerics.golden.GoldenIo.section
import numerics.golden.GoldenIo.toDbl
import numerics.golden.GoldenIo.toVec
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** Эталон `solve.json`: `solve`, `relativeBackwardError`, `solveDiagnosed().forwardError`. */
@Tag("fast")
class GoldenSolveTest {

    @TestFactory
    fun cases(): List<DynamicTest> {
        val golden = section(GoldenIo.readGolden("solve.json"), "cases")
        return GoldenInputs.solveCases.map { case ->
            DynamicTest.dynamicTest("solve ${case.key}") {
                val exp = obj(entry(golden, case.key))
                val tol = GoldenInputs.tolFor(case)
                val a = case.matrix()
                val b = GoldenInputs.vec(case.n)

                val x = LinearAlgebra.solve(a, b)
                assertClose(toVec(exp["x"]), x, tol, "${case.key}.x")

                val omega = Conditioning.relativeBackwardError(a, b, x)
                assertClose(toDbl(exp["omega"]), omega, tol, "${case.key}.omega")

                val fe = LinearAlgebra.solveDiagnosed(a, b).forwardError
                assertForwardErrorClose(obj(exp["forwardError"]), fe, tol, "${case.key}.forwardError")
            }
        }
    }

    companion object {
        /** Тип — точно; числовые поля — относительно `tol` (зависят от LAPACK). */
        fun assertForwardErrorClose(exp: Map<String, Any?>, got: ForwardError, tol: Double, label: String) {
            val expType = exp["type"] as String
            val gotType = when (got) {
                is ForwardError.Bounded -> "Bounded"
                is ForwardError.NoFiniteBound -> "NoFiniteBound"
                is ForwardError.Unreliable -> "Unreliable"
            }
            assertEquals(expType, gotType, "$label.type")
            assertClose(toDbl(exp["backwardError"]), got.backwardError, tol, "$label.backwardError")
            when (got) {
                is ForwardError.Bounded -> {
                    assertClose(toDbl(exp["cond"]), got.cond, tol, "$label.cond")
                    assertClose(toDbl(exp["relativeBound"]), got.relativeBound, tol, "$label.relativeBound")
                }
                is ForwardError.NoFiniteBound -> Unit
                is ForwardError.Unreliable -> {
                    val c = obj(exp["condition"])
                    assertClose(toDbl(c["condInf"]), got.condition.condInf, tol, "$label.condition.condInf")
                    assertClose(
                        toDbl(c["inversionResidual"]),
                        got.condition.inversionResidual,
                        tol,
                        "$label.condition.inversionResidual",
                    )
                    assertClose(toDbl(c["tolerance"]), got.condition.tolerance, tol, "$label.condition.tolerance")
                    assertEquals(c["isReliable"] as Boolean, got.condition.isReliable, "$label.condition.isReliable")
                }
            }
        }
    }
}
