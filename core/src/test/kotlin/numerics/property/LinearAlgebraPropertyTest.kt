package numerics.property

import net.jqwik.api.Arbitraries
import net.jqwik.api.Arbitrary
import net.jqwik.api.Assume
import net.jqwik.api.ForAll
import net.jqwik.api.Property
import net.jqwik.api.Provide
import net.jqwik.api.Tag
import numerics.Conditioning
import numerics.DenseMatrix
import numerics.LinearAlgebra
import numerics.ReferenceOracle
import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.oracle.HipparchusOracle
import numerics.property.PropertySupport.absMatVec
import numerics.property.PropertySupport.maxAbs
import numerics.property.PropertySupport.norm1
import numerics.property.PropertySupport.normInf
import numerics.property.PropertySupport.plainMatVec
import numerics.property.PropertySupport.relDiff
import numerics.property.PropertySupport.residualToIdentity
import numerics.property.PropertySupport.sub
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import kotlin.math.abs
import kotlin.math.max

@Tag("fast")
class LinearAlgebraPropertyTest {

    @Provide
    fun backends(): Arbitrary<LinAlgBackend> =
        Arbitraries.of(listOfNotNull(Backends.java(), if (Backends.isNativeAvailable()) Backends.native() else null))

    @Provide fun systems(): Arbitrary<Pair<DenseMatrix, DoubleArray>> = Generators.systemGeneral()
    @Provide fun squares(): Arbitrary<DenseMatrix> = Generators.squareGeneral()
    @Provide fun spd(): Arbitrary<DenseMatrix> = Generators.spd()
    @Provide fun scales(): Arbitrary<Double> = Arbitraries.of(1e-8, 1e8)

    @Provide
    fun weighted(): Arbitrary<Pair<DenseMatrix, DoubleArray>> =
        Generators.squareGeneral().flatMap { a -> Generators.vector(a.rows).map { w -> a to DoubleArray(w.size) { abs(w[it]) + 0.5 } } }

    private fun cond(a: DenseMatrix, backend: LinAlgBackend) = Conditioning.conditionEstimate(a, backend = backend).condInf

    @Property(tries = 100)
    fun `solve residual is bounded by the system scale`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val x = LinearAlgebra.solve(a, b, backend)
        val r = maxAbs(sub(plainMatVec(a, x), b))
        val bound = 1e-10 * (normInf(a) * maxAbs(x) + maxAbs(b))
        assertTrue(r <= bound) { "residual $r > $bound (n=${a.rows}, backend=$backend)" }
    }

    @Property(tries = 100)
    fun `solve agrees with Hipparchus within cond`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val c = cond(a, backend)
        Assume.that(c < 1e10)
        val d = relDiff(LinearAlgebra.solve(a, b, backend), HipparchusOracle.solve(a, b))
        assertTrue(d <= 1e-8 * c) { "difference from Hipparchus $d > ${1e-8 * c} (cond=$c, n=${a.rows})" }
    }

    @Property(tries = 100)
    fun `solve agrees with ReferenceOracle within cond`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val c = cond(a, backend)
        Assume.that(c < 1e10)
        val d = relDiff(LinearAlgebra.solve(a, b, backend), ReferenceOracle.solve(a.toRows(), b))
        assertTrue(d <= 1e-8 * c) { "difference from ReferenceOracle $d > ${1e-8 * c} (cond=$c, n=${a.rows})" }
    }

    @Property(tries = 100)
    fun `multiplication by the identity is a bit-for-bit no-op`(@ForAll("squares") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val id = DenseMatrix.identity(a.rows)
        assertTrue(LinearAlgebra.matMat(a, id, backend).data.contentEquals(a.data)) { "A·I ≠ A (n=${a.rows})" }
        assertTrue(LinearAlgebra.matMat(id, a, backend).data.contentEquals(a.data)) { "I·A ≠ A (n=${a.rows})" }
    }

    @Property(tries = 100)
    fun `matTransVec matches matVec of the transpose`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, y) = s
        val at = a.transpose()
        val d = maxAbs(sub(LinearAlgebra.matTransVec(a, y, backend), LinearAlgebra.matVec(at, y, backend)))
        val scale = max(maxAbs(absMatVec(at, y)), 1.0)
        assertTrue(d <= 1e-13 * scale) { "matTransVec vs matVec(Aᵀ): $d > ${1e-13 * scale}" }
    }

    @Property(tries = 100)
    fun `atWa matches the explicit Aᵀ·(W·A)`(@ForAll("weighted") s: Pair<DenseMatrix, DoubleArray>, @ForAll("backends") backend: LinAlgBackend) {
        val (a, w) = s
        val scaledRows = DenseMatrix.build(a.rows, a.cols) { i, j -> w[i] * a[i, j] }
        val expected = LinearAlgebra.matMat(a.transpose(), scaledRows, backend)
        val d = maxAbs(sub(LinearAlgebra.atWa(a, w, backend).data, expected.data))
        val scale = max(maxAbs(w) * norm1(a) * norm1(a), 1.0)
        assertTrue(d <= 1e-12 * scale) { "atWa: $d > ${1e-12 * scale}" }
    }

    @Property(tries = 100)
    fun `solve is invariant to a common scaling of the system`(@ForAll("systems") s: Pair<DenseMatrix, DoubleArray>, @ForAll("scales") c: Double, @ForAll("backends") backend: LinAlgBackend) {
        val (a, b) = s
        val k = cond(a, backend)
        Assume.that(k < 1e10)
        val x = LinearAlgebra.solve(a, b, backend)
        val xs = LinearAlgebra.solve(Generators.scale(a, c), Generators.scale(b, c), backend)
        // two independent solutions differ by about cond·eps — the tolerance is proportional to cond
        val d = relDiff(xs, x)
        assertTrue(d <= 1e-12 * max(k, 1.0)) { "scale $c: $d > ${1e-12 * max(k, 1.0)} (cond=$k)" }
    }

    @Property(tries = 100)
    fun `inverse gives two-sided residuals within cond`(@ForAll("squares") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val c = cond(a, backend)
        Assume.that(c < 1e8)
        val x = Conditioning.inverse(a, backend)
        assertNotNull(x) { "inverse returned null at cond=$c" }
        val right = residualToIdentity(LinearAlgebra.matMat(a, x!!, backend))
        val left = residualToIdentity(LinearAlgebra.matMat(x, a, backend))
        assertTrue(right <= 1e-8 * c) { "‖A·X − I‖∞ = $right > ${1e-8 * c}" }
        assertTrue(left <= 1e-8 * c) { "‖X·A − I‖∞ = $left > ${1e-8 * c}" }
    }

    @Property(tries = 100)
    fun `cholesky reconstructs the SPD matrix and agrees with Hipparchus`(@ForAll("spd") a: DenseMatrix, @ForAll("backends") backend: LinAlgBackend) {
        val l = LinearAlgebra.cholesky(a, backend)
        assertNotNull(l) { "cholesky returned null for an SPD matrix" }
        val n = a.rows
        for (i in 0 until n) {
            assertTrue(l!![i, i] > 0.0) { "L[$i,$i] = ${l[i, i]} is not positive" }
            for (j in i + 1 until n) assertTrue(l[i, j] == 0.0) { "L[$i,$j] = ${l[i, j]} ≠ 0 above the diagonal" }
        }
        val d = relDiff(LinearAlgebra.matMat(l!!, l.transpose(), backend), a)
        assertTrue(d <= 1e-12) { "L·Lᵀ vs A: $d" }
        val h = relDiff(l, HipparchusOracle.cholesky(a))
        assertTrue(h <= 1e-10) { "L vs Hipparchus: $h" }
    }
}
