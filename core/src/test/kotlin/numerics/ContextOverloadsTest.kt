package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.DynamicTest.dynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Перегрузки с [NumericsContext] побитово совпадают с перегрузками, принимающими бэкенд напрямую. */
@Tag("fast")
class ContextOverloadsTest {

    private fun spd(n: Int, seed: Long): DenseMatrix {
        val g = java.util.Random(seed)
        val b = DenseMatrix.build(n, n) { _, _ -> g.nextDouble() - 0.5 }
        return LinearAlgebra.addScaled(LinearAlgebra.matMat(b.transpose(), b), DenseMatrix.identity(n), n.toDouble())
    }

    private fun bits(x: DoubleArray) = x.map { it.toRawBits() }
    private fun bits(x: DenseMatrix) = x.rows to x.cols to bits(x.data)

    @TestFactory
    fun `перегрузки с контекстом равны перегрузкам с бэкендом`(): List<DynamicTest> =
        Backends.available().flatMap { backend -> cases(backend, NumericsContext(backend = backend)) }

    private fun cases(be: LinAlgBackend, ctx: NumericsContext): List<DynamicTest> {
        val a = spd(5, 11); val b = spd(5, 12)
        val v = DoubleArray(5) { 0.3 * it - 1.0 }
        val w = DoubleArray(5) { 1.0 + it }
        val ops = listOf<Pair<String, Pair<() -> Any, () -> Any>>>(
            "matVec" to ({ bits(LinearAlgebra.matVec(a, v, be)) } to { bits(LinearAlgebra.matVec(a, v, ctx)) }),
            "matTransVec" to ({ bits(LinearAlgebra.matTransVec(a, v, be)) } to { bits(LinearAlgebra.matTransVec(a, v, ctx)) }),
            "matMat" to ({ bits(LinearAlgebra.matMat(a, b, be)) } to { bits(LinearAlgebra.matMat(a, b, ctx)) }),
            "atWa" to ({ bits(LinearAlgebra.atWa(a, w, be)) } to { bits(LinearAlgebra.atWa(a, w, ctx)) }),
            "addScaled" to ({ bits(LinearAlgebra.addScaled(a, b, 0.5, be)) } to { bits(LinearAlgebra.addScaled(a, b, 0.5, ctx)) }),
            "solve" to ({ bits(LinearAlgebra.solve(a, v, be)) } to { bits(LinearAlgebra.solve(a, v, ctx)) }),
            "solveDiagnosed" to ({
                LinearAlgebra.solveDiagnosed(a, v, be).let { bits(it.x) to it.forwardError.toString() }
            } to { LinearAlgebra.solveDiagnosed(a, v, ctx).let { bits(it.x) to it.forwardError.toString() } }),
            "solveDiagnosed(ESTIMATE)" to ({
                LinearAlgebra.solveDiagnosed(a, v, be, ConditionSource.ESTIMATE).forwardError.toString()
            } to { LinearAlgebra.solveDiagnosed(a, v, ctx, ConditionSource.ESTIMATE).forwardError.toString() }),
            "cholesky" to ({ bits(LinearAlgebra.cholesky(a, be)!!) } to { bits(LinearAlgebra.cholesky(a, ctx)!!) }),
            "inverse" to ({ bits(Conditioning.inverse(a, be)!!) } to { bits(Conditioning.inverse(a, ctx)!!) }),
            "conditionInf" to ({ Conditioning.conditionInf(a, backend = be) } to { Conditioning.conditionInf(a, ctx) }),
            "conditionEstimate" to ({ Conditioning.conditionEstimate(a, backend = be) } to { Conditioning.conditionEstimate(a, ctx) }),
            "symmetricEigenvalues" to ({ bits(Conditioning.symmetricEigenvalues(a, be)) } to { bits(Conditioning.symmetricEigenvalues(a, ctx)) }),
            "conditionSymmetric" to ({ Conditioning.conditionSymmetric(a, be).toRawBits() } to { Conditioning.conditionSymmetric(a, ctx).toRawBits() }),
        )
        return ops.map { (name, pair) ->
            dynamicTest("$name [${be.name}]") { assertEquals(pair.first(), pair.second(), name) }
        }
    }

    @Test
    fun `inverse через контекст возвращает null на вырожденной, а инверсия вырожденной и NaN отсекается`() {
        val singular = DenseMatrix.fromRows(arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0)))
        assertNull(Conditioning.inverse(singular, NumericsContext()))
        assertEquals(Double.POSITIVE_INFINITY, Conditioning.conditionEstimate(singular).condInf)
        assertEquals(Double.POSITIVE_INFINITY, Conditioning.conditionInf(singular).condInf)
        val tol = assertFailsWith<IllegalArgumentException> { Conditioning.conditionEstimate(singular, tolerance = 0.0) }
        assertTrue(tol.message!!.contains("tolerance"), tol.message)
        val huge = DenseMatrix.diagonal(doubleArrayOf(1e-320, 1.0))
        assertNull(Conditioning.inverse(huge), "обращение с переполнением — нечисловое, а не мусор")
        val r = Conditioning.inversionResidual(
            arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 4.0)),
            arrayOf(doubleArrayOf(0.5, 0.0), doubleArrayOf(0.0, 0.25)),
        )
        assertEquals(0.0, r)
        val e = assertFailsWith<IllegalArgumentException> { ConditionEstimate(1.0, 0.0, 0.0) }
        assertTrue(e.message!!.contains("tolerance"), e.message)
        val nan = assertFailsWith<IllegalArgumentException> { Conditioning.forwardError(ConditionEstimate(1.0, 0.0, 1e-8), Double.NaN) }
        assertTrue(nan.message!!.contains("конечной"), nan.message)
    }

    @Test
    fun `адаптеры над массивами строк проверяют пустую B и число строк`() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        assertTrue(assertFailsWith<IllegalArgumentException> { LinearAlgebra.matMat(a, arrayOf(DoubleArray(0))) }.message!!.contains("пустая"))
        assertTrue(assertFailsWith<IllegalArgumentException> { LinearAlgebra.matMat(a, emptyArray()) }.message!!.contains("пустая"))
        assertTrue(assertFailsWith<IllegalArgumentException> { LinearAlgebra.addScaled(a, arrayOf(doubleArrayOf(1.0, 2.0)), 1.0) }.message!!.contains("строк"))
    }

    @Test
    fun `сборка через контекст с отдельным пулом совпадает с последовательной`() {
        val n = 37
        val ctx = NumericsContext(parallel = true, parallelism = 2)
        val seq = ParallelAssembly.assembleMatrix(n, n, parallel = false) { i, j -> 1.0 / (i + j + 1) }
        val par = ParallelAssembly.assembleMatrix(n, n, ctx) { i, j -> 1.0 / (i + j + 1) }
        for (i in 0 until n) assertContentEquals(seq[i], par[i])
        val rowsSeq = ParallelAssembly.assembleRows(n, 3, parallel = false) { i -> DoubleArray(3) { i * 3.0 + it } }
        val rowsPar = ParallelAssembly.assembleRows(n, 3, ctx) { i -> DoubleArray(3) { i * 3.0 + it } }
        for (i in 0 until n) assertContentEquals(rowsSeq[i], rowsPar[i])
        val dense = ParallelAssembly.assembleDense(n, n, ctx) { i, j -> 1.0 / (i + j + 1) }
        assertEquals(DenseMatrix.fromRows(seq), dense)
        val e = assertFailsWith<IllegalArgumentException> { ParallelAssembly.assembleRows(-1, 2, ctx) { DoubleArray(2) } }
        assertTrue(e.message!!.contains("отрицательными"), e.message)
    }
}
