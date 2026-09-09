package numerics.golden

import numerics.LinearAlgebra
import numerics.golden.GoldenIo.assertClose
import numerics.golden.GoldenIo.entry
import numerics.golden.GoldenIo.obj
import numerics.golden.GoldenIo.section
import numerics.golden.GoldenIo.toDbl
import numerics.golden.GoldenIo.toMat
import numerics.golden.GoldenIo.toVec
import numerics.golden.GoldenInputs.DEFAULT_TOL
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory

/** Эталон `matops.json`: матрично-векторные операции и нормы. */
@Tag("fast")
class GoldenMatOpsTest {

    @TestFactory
    fun cases(): List<DynamicTest> {
        val golden = section(GoldenIo.readGolden("matops.json"), "cases")
        return GoldenInputs.matOpsCases.flatMap { case ->
            val exp = obj(entry(golden, case.key))
            val a = case.matrix()
            val b = GoldenInputs.vec(case.n)
            val aT = GoldenInputs.transpose(a)
            val w = GoldenInputs.posVec(case.n)
            val k = case.key
            listOf(
                DynamicTest.dynamicTest("matVec $k") {
                    assertClose(toVec(exp["matVec"]), LinearAlgebra.matVec(a, b), DEFAULT_TOL, "$k.matVec")
                },
                DynamicTest.dynamicTest("matTransVec $k") {
                    assertClose(toVec(exp["matTransVec"]), LinearAlgebra.matTransVec(a, b), DEFAULT_TOL, "$k.matTransVec")
                },
                DynamicTest.dynamicTest("matMat $k") {
                    assertClose(toMat(exp["matMat"]), LinearAlgebra.matMat(a, a), DEFAULT_TOL, "$k.matMat")
                },
                DynamicTest.dynamicTest("atWa $k") {
                    assertClose(toMat(exp["atWa"]), LinearAlgebra.atWa(a, w), DEFAULT_TOL, "$k.atWa")
                },
                DynamicTest.dynamicTest("addScaled $k") {
                    assertClose(toMat(exp["addScaled"]), LinearAlgebra.addScaled(a, aT, 0.5), DEFAULT_TOL, "$k.addScaled")
                },
                DynamicTest.dynamicTest("norm2 $k") {
                    assertClose(toDbl(exp["norm2"]), LinearAlgebra.norm2(b), DEFAULT_TOL, "$k.norm2")
                },
                DynamicTest.dynamicTest("normInf $k") {
                    assertClose(toDbl(exp["normInf"]), LinearAlgebra.normInf(b), DEFAULT_TOL, "$k.normInf")
                },
                DynamicTest.dynamicTest("maxAsymmetry $k") {
                    assertClose(toDbl(exp["maxAsymmetry"]), LinearAlgebra.maxAsymmetry(a), DEFAULT_TOL, "$k.maxAsymmetry")
                },
            )
        }
    }
}
