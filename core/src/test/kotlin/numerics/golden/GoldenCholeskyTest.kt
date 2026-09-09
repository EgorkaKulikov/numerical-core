package numerics.golden

import numerics.LinearAlgebra
import numerics.golden.GoldenIo.assertClose
import numerics.golden.GoldenIo.entry
import numerics.golden.GoldenIo.obj
import numerics.golden.GoldenIo.section
import numerics.golden.GoldenIo.toMat
import numerics.golden.GoldenInputs.DEFAULT_TOL
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertNotNull

/** Эталон `cholesky.json`: нижний треугольный множитель `L` для SPD-матриц. */
@Tag("fast")
class GoldenCholeskyTest {

    @TestFactory
    fun cases(): List<DynamicTest> {
        val golden = section(GoldenIo.readGolden("cholesky.json"), "cases")
        return GoldenInputs.choleskyCases.map { case ->
            DynamicTest.dynamicTest("cholesky ${case.key}") {
                val exp = obj(entry(golden, case.key))
                val l = LinearAlgebra.cholesky(case.matrix())
                assertNotNull(l, "${case.key}: cholesky вернул null")
                assertClose(toMat(exp["L"]), l, DEFAULT_TOL, "${case.key}.L")
            }
        }
    }
}
