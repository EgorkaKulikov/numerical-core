package numerics.property

import net.jqwik.api.Arbitraries
import net.jqwik.api.Arbitrary
import net.jqwik.api.Assume
import net.jqwik.api.ForAll
import net.jqwik.api.Property
import net.jqwik.api.Provide
import net.jqwik.api.Tag
import numerics.ConditionSource
import numerics.Conditioning
import numerics.DenseMatrix
import numerics.ForwardError
import numerics.LinearAlgebra
import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.oracle.HipparchusOracle
import numerics.property.PropertySupport.normInf
import numerics.property.PropertySupport.relDiff
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import kotlin.math.abs
import kotlin.math.max

@Tag("fast")
class ConditioningPropertyTest {

    @Provide
    fun backends(): Arbitrary<LinAlgBackend> =
        Arbitraries.of(listOfNotNull(Backends.java(), if (Backends.isNativeAvailable()) Backends.native() else null))

    @Provide fun systems(): Arbitrary<Pair<DenseMatrix, DoubleArray>> = Generators.systemGeneral()
    @Provide fun squares(): Arbitrary<DenseMatrix> = Generators.squareGeneral()
    @Provide fun symmetric(): Arbitrary<DenseMatrix> = Generators.symmetric()
    @Provide fun spd(): Arbitrary<DenseMatrix> = Generators.spd()
    @Provide fun spdSystems(): Arbitrary<Pair<DenseMatrix, DoubleArray>> = Generators.spd().flatMap { a -> Generators.vector(a.rows).map { a to it } }
    @Provide fun omegas(): Arbitrary<Double> = Arbitraries.of(0.0, 1e-16, 1e-10)
    @Provide fun sources(): Arbitrary<ConditionSource> = Arbitraries.of(ConditionSource.INVERSION, ConditionSource.ESTIMATE)

    @Property(tries = 100)
    fun `сумма собственных значений равна следу и совпадает с Hipparchus`(@ForAll("symmetric") s: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val lambda = Conditioning.symmetricEigenvalues(s, backend)
        var trace = 0.0
        for (i in 0 until s.rows) trace += s[i, i]
        val bound = 1e-10 * max(normInf(s), 1.0)
        assertTrue(abs(lambda.sum() - trace) <= bound) { "Σλ − tr = ${lambda.sum() - trace} > $bound" }
        val d = relDiff(lambda.sortedArray(), HipparchusOracle.symmetricEigenvalues(s).sortedArray())
        assertTrue(d <= 1e-10) { "λ vs Hipparchus: $d" }
    }

    @Property(tries = 100)
    fun `SPD имеет положительный спектр и conditionSymmetric равен λmax÷λmin`(@ForAll("spd") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val lambda = Conditioning.symmetricEigenvalues(a, backend)
        for (v in lambda) assertTrue(v > 0.0) { "λ = $v ≤ 0 для SPD" }
        val ref = HipparchusOracle.symmetricEigenvalues(a)
        val expected = ref.max() / ref.min()
        val got = Conditioning.conditionSymmetric(a, backend)
        assertTrue(abs(got - expected) <= 1e-9 * expected) { "conditionSymmetric $got vs $expected" }
    }

    @Property(tries = 100)
    fun `число обусловленности не меньше единицы`(@ForAll("squares") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        assertTrue(Conditioning.conditionInf(a, backend = backend).condInf >= 1.0 - 1e-12)
        assertTrue(Conditioning.conditionEstimate(a, backend = backend).condInf >= 1.0 - 1e-12)
    }

    @Property(tries = 100)
    fun `оценка cond∞ лежит в окне относительно точного cond₁`(@ForAll("squares") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val cond1 = HipparchusOracle.condition1(a)
        Assume.that(cond1 < 1e10)
        val ratio = Conditioning.conditionEstimate(a, backend = backend).condInf / cond1
        assertTrue(ratio >= 0.01 && ratio <= 1.0001) { "condEst/cond1 = $ratio вне [0.01, 1.0001] (n=${a.rows})" }
    }

    @Property(tries = 100)
    fun `обратная ошибка решения на уровне машинной точности`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val omega = Conditioning.relativeBackwardError(a, b, LinearAlgebra.solve(a, b, backend), backend)
        assertTrue(omega <= 1e-12) { "ω = $omega" }
    }

    @Property(tries = 100)
    fun `forwardError ограничен тогда и только тогда, когда оценка достоверна`(@ForAll("squares") a: DenseMatrix, @ForAll("omegas") omega: Double, @ForAll("backends") backend: LinAlgBackend) {
        val est = Conditioning.conditionInf(a, backend = backend)
        val fe = Conditioning.forwardError(est, omega)
        assertEquals(est.isReliable, fe is ForwardError.Bounded) { "isReliable=${est.isReliable}, forwardError=$fe" }
        if (fe is ForwardError.Bounded) {
            assertEquals((est.condInf * omega).toRawBits(), fe.relativeBound.toRawBits())
        }
    }

    @Property(tries = 100)
    fun `solveDiagnosed возвращает то же x, что и solve`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("sources") source: ConditionSource, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val x = LinearAlgebra.solve(a, b, backend)
        assertTrue(LinearAlgebra.solveDiagnosed(a, b, backend, source).x.contentEquals(x)) { "x расходится для $source" }
    }

    @Property(tries = 100)
    fun `solveDiagnosed по спектру возвращает то же x на SPD`(@ForAll("spdSystems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val x = LinearAlgebra.solve(a, b, backend)
        assertTrue(LinearAlgebra.solveDiagnosed(a, b, backend, ConditionSource.SYMMETRIC_SPECTRUM).x.contentEquals(x))
    }
}
