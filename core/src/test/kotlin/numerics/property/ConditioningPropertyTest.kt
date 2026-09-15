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
    fun `eigenvalues sum to the trace and match Hipparchus`(@ForAll("symmetric") s: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val lambda = Conditioning.symmetricEigenvalues(s, backend)
        var trace = 0.0
        for (i in 0 until s.rows) trace += s[i, i]
        val bound = 1e-10 * max(normInf(s), 1.0)
        assertTrue(abs(lambda.sum() - trace) <= bound) { "Σλ − tr = ${lambda.sum() - trace} > $bound" }
        val d = relDiff(lambda.sortedArray(), HipparchusOracle.symmetricEigenvalues(s).sortedArray())
        assertTrue(d <= 1e-10) { "λ vs Hipparchus: $d" }
    }

    @Property(tries = 100)
    fun `SPD has a positive spectrum and conditionSymmetric equals λmax÷λmin`(@ForAll("spd") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val lambda = Conditioning.symmetricEigenvalues(a, backend)
        for (v in lambda) assertTrue(v > 0.0) { "λ = $v ≤ 0 for an SPD matrix" }
        val ref = HipparchusOracle.symmetricEigenvalues(a)
        val expected = ref.max() / ref.min()
        val got = Conditioning.conditionSymmetric(a, backend)
        assertTrue(abs(got - expected) <= 1e-9 * expected) { "conditionSymmetric $got vs $expected" }
    }

    @Property(tries = 100)
    fun `condition number is at least one`(@ForAll("squares") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        assertTrue(Conditioning.conditionInf(a, backend = backend).condInf >= 1.0 - 1e-12)
        assertTrue(Conditioning.conditionEstimate(a, backend = backend).condInf >= 1.0 - 1e-12)
    }

    @Property(tries = 100)
    fun `cond∞ estimate lies within a window of the exact cond₁`(@ForAll("squares") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val cond1 = HipparchusOracle.condition1(a)
        Assume.that(cond1 < 1e10)
        val ratio = Conditioning.conditionEstimate(a, backend = backend).condInf / cond1
        assertTrue(ratio >= 0.01 && ratio <= 1.0001) { "condEst/cond1 = $ratio outside [0.01, 1.0001] (n=${a.rows})" }
    }

    @Property(tries = 100)
    fun `backward error of the solution is at machine precision`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val omega = Conditioning.relativeBackwardError(a, b, LinearAlgebra.solve(a, b, backend), backend)
        assertTrue(omega <= 1e-12) { "ω = $omega" }
    }

    @Property(tries = 100)
    fun `forwardError is bounded iff the estimate is reliable`(@ForAll("squares") a: DenseMatrix, @ForAll("omegas") omega: Double, @ForAll("backends") backend: LinAlgBackend) {
        val est = Conditioning.conditionInf(a, backend = backend)
        val fe = Conditioning.forwardError(est, omega)
        assertEquals(est.isReliable, fe is ForwardError.Bounded) { "isReliable=${est.isReliable}, forwardError=$fe" }
        if (fe is ForwardError.Bounded) {
            assertEquals((est.condInf * omega).toRawBits(), fe.relativeBound.toRawBits())
        }
    }

    @Property(tries = 100)
    fun `solveDiagnosed returns the same x as solve`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("sources") source: ConditionSource, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val x = LinearAlgebra.solve(a, b, backend)
        assertTrue(LinearAlgebra.solveDiagnosed(a, b, backend, source).x.contentEquals(x)) { "x differs for $source" }
    }

    @Property(tries = 100)
    fun `solveDiagnosed via spectrum returns the same x on SPD`(@ForAll("spdSystems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val x = LinearAlgebra.solve(a, b, backend)
        assertTrue(LinearAlgebra.solveDiagnosed(a, b, backend, ConditionSource.SYMMETRIC_SPECTRUM).x.contentEquals(x))
    }
}
