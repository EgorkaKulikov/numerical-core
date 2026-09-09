package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import java.util.Random
import kotlin.math.abs
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Обусловленность и разложение Холецкого через LAPACK-примитивы обоих бэкендов. */
@Tag("fast")
class ConditioningLapackTest {

    private val backends: List<Pair<String, () -> LinAlgBackend>> = listOf(
        "java" to { Backends.java() },
        "native" to {
            Assumptions.assumeTrue(Backends.isNativeAvailable(), "нативная BLAS/LAPACK недоступна")
            Backends.native()
        },
    )

    /** Диагонально доминирующая симметричная положительно определённая матрица. */
    private fun dd(n: Int): DenseMatrix = DenseMatrix.build(n, n) { i, j -> if (i == j) n + 1.0 else 1.0 / (1.0 + i + j) }

    private fun mat(vararg rows: DoubleArray): DenseMatrix = DenseMatrix.fromRows(arrayOf(*rows))

    private fun cases(name: String, body: (LinAlgBackend) -> Unit): List<DynamicTest> =
        backends.map { (label, make) -> DynamicTest.dynamicTest("$name [$label]") { body(make()) } }

    @TestFactory
    fun choleskyRejectsMalformed(): List<DynamicTest> = cases("cholesky: несимметричная, неквадратная и NaN → IAE") { be ->
        assertFailsWith<IllegalArgumentException> {
            LinearAlgebra.cholesky(mat(doubleArrayOf(4.0, 100.0), doubleArrayOf(2.0, 3.0)), be)
        }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.cholesky(DenseMatrix.zeros(2, 3), be) }
        assertFailsWith<IllegalArgumentException> {
            LinearAlgebra.cholesky(mat(doubleArrayOf(Double.NaN, 0.0), doubleArrayOf(0.0, 1.0)), be)
        }
    }

    @TestFactory
    fun choleskyIndefiniteIsNull(): List<DynamicTest> = cases("cholesky: индефинитная → null") { be ->
        assertNull(LinearAlgebra.cholesky(mat(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 1.0)), be))
    }

    @TestFactory
    fun choleskyReconstructsSpd(): List<DynamicTest> = cases("cholesky: L·Lᵀ ≈ A, верхний треугольник нулевой") { be ->
        val a = dd(8)
        val l = assertNotNull(LinearAlgebra.cholesky(a, be))
        for (i in 0 until 8) for (j in i + 1 until 8) assertEquals(0.0, l[i, j])
        val llt = LinearAlgebra.matMat(l, l.transpose(), be)
        for (i in 0 until 8) for (j in 0 until 8) assertEquals(a[i, j], llt[i, j], 1e-12, "($i,$j)")
    }

    @TestFactory
    fun symmetricEigenvalues(): List<DynamicTest> = cases("symmetricEigenvalues: контракт и след") { be ->
        assertFailsWith<IllegalArgumentException> {
            Conditioning.symmetricEigenvalues(mat(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 1.0)), be)
        }
        val eig = Conditioning.symmetricEigenvalues(DenseMatrix.diagonal(doubleArrayOf(3.0, 1.0, 2.0)), be)
        assertContentEquals(doubleArrayOf(1.0, 2.0, 3.0), eig)
        val n = 16
        val rnd = Random(42)
        val s = DenseMatrix.zeros(n, n)
        for (i in 0 until n) for (j in i until n) {
            val v = rnd.nextDouble() * 2.0 - 1.0
            s[i, j] = v
            s[j, i] = v
        }
        var trace = 0.0
        for (i in 0 until n) trace += s[i, i]
        assertEquals(trace, Conditioning.symmetricEigenvalues(s, be).sum(), 1e-10)
    }

    @TestFactory
    fun conditionEstimate(): List<DynamicTest> = cases("conditionEstimate: единичная, вырожденная, отношение к conditionInf") { be ->
        val one = Conditioning.conditionEstimate(DenseMatrix.identity(5), backend = be)
        assertTrue(one.condInf in (1.0 - 1e-12)..(1.0 + 1e-12), "cond(I) = ${one.condInf}")
        assertTrue(one.isReliable)
        val singular = Conditioning.conditionEstimate(mat(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)), backend = be)
        assertEquals(Double.POSITIVE_INFINITY, singular.condInf)
        assertEquals(0.0, singular.inversionResidual)
        assertNull(singular.valueOrNull())
        val a = dd(16)
        val ratio = Conditioning.conditionEstimate(a, backend = be).condInf / Conditioning.conditionInf(a, backend = be).condInf
        assertTrue(ratio in 0.05..20.0, "отношение оценок = $ratio")
    }

    @TestFactory
    fun inverse(): List<DynamicTest> = cases("inverse: единичная и вырожденная") { be ->
        val inv = assertNotNull(Conditioning.inverse(DenseMatrix.identity(4), be))
        for (i in 0 until 4) for (j in 0 until 4) assertEquals(if (i == j) 1.0 else 0.0, inv[i, j], 1e-15)
        assertNull(Conditioning.inverse(mat(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)), be))
    }

    @TestFactory
    fun solveDiagnosedEstimate(): List<DynamicTest> = cases("solveDiagnosed(ESTIMATE): Bounded и то же x, что у solve") { be ->
        val a = dd(8)
        val b = DoubleArray(8) { 1.0 + it }
        val d = LinearAlgebra.solveDiagnosed(a, b, be, source = ConditionSource.ESTIMATE)
        val fe = assertIs<ForwardError.Bounded>(d.forwardError)
        assertTrue(fe.cond >= 1.0 && fe.relativeBound >= 0.0 && abs(fe.cond * fe.backwardError - fe.relativeBound) == 0.0)
        assertContentEquals(LinearAlgebra.solve(a, b, be), d.x)
    }
}
