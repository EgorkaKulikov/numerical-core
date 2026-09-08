package numerics.golden

import numerics.Conditioning
import numerics.golden.GoldenIo.assertClose
import numerics.golden.GoldenIo.entry
import numerics.golden.GoldenIo.obj
import numerics.golden.GoldenIo.section
import numerics.golden.GoldenIo.toDbl
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** Эталон `conditioning.json`: `conditionInf` — `condInf`, `inversionResidual`, `isReliable`. */
@Tag("fast")
class GoldenConditioningTest {

    @TestFactory
    fun cases(): List<DynamicTest> {
        val golden = section(GoldenIo.readGolden("conditioning.json"), "cases")
        return GoldenInputs.conditioningCases.map { case ->
            DynamicTest.dynamicTest("conditionInf ${case.key}") {
                val exp = obj(entry(golden, case.key))
                val tol = GoldenInputs.tolFor(case)
                val est = Conditioning.conditionInf(case.matrix())
                assertClose(toDbl(exp["condInf"]), est.condInf, tol, "${case.key}.condInf")
                assertClose(toDbl(exp["inversionResidual"]), est.inversionResidual, tol, "${case.key}.inversionResidual")
                assertEquals(exp["isReliable"] as Boolean, est.isReliable, "${case.key}.isReliable")
            }
        }
    }
}
