package numerics.golden

import numerics.Conditioning
import numerics.golden.GoldenIo.assertClose
import numerics.golden.GoldenIo.entry
import numerics.golden.GoldenIo.section
import numerics.golden.GoldenIo.toDbl
import numerics.golden.GoldenIo.toVec
import numerics.golden.GoldenInputs.DEFAULT_TOL
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory

/** Эталон `eigen.json`: `symmetricEigenvalues` (отсортированные) и `conditionSymmetric`. */
@Tag("fast")
class GoldenEigenTest {

    @TestFactory
    fun cases(): List<DynamicTest> {
        val root = GoldenIo.readGolden("eigen.json")
        val eig = section(root, "symmetricEigenvalues")
        val cond = section(root, "conditionSymmetric")
        val eigTests = GoldenInputs.eigenCases.map { case ->
            DynamicTest.dynamicTest("symmetricEigenvalues ${case.key}") {
                val got = Conditioning.symmetricEigenvalues(case.matrix()).sortedArray()
                assertClose(toVec(entry(eig, case.key)), got, DEFAULT_TOL, "${case.key}.eigenvalues")
            }
        }
        val condTests = GoldenInputs.conditionSymmetricCases.map { case ->
            DynamicTest.dynamicTest("conditionSymmetric ${case.key}") {
                val got = Conditioning.conditionSymmetric(case.matrix())
                assertClose(toDbl(entry(cond, case.key)), got, DEFAULT_TOL, "${case.key}.conditionSymmetric")
            }
        }
        return eigTests + condTests
    }
}
