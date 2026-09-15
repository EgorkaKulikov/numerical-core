package numerics

import numerics.backend.Backends
import numerics.backend.LuFactorization
import numerics.backend.MatrixNorm
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import kotlin.math.sqrt
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

/** Contracts not covered by other test classes: DenseMatrix, Shapes, Backends, NetlibBackend. */
@Tag("fast")
class CoverageGapsTest {

    private fun m(vararg rows: DoubleArray) = DenseMatrix.fromRows(arrayOf(*rows))
    private fun r(vararg v: Double) = doubleArrayOf(*v)
    private fun rnd(n: Int, c: Int, seed: Long) = java.util.Random(seed).let { g ->
        DenseMatrix.build(n, c) { i, j -> g.nextDouble() - 0.5 + if (i == j) n.toDouble() else 0.0 }
    }

    @Test
    fun `DenseMatrix toString shows dimensions and ellipses only when truncated`() {
        val small = m(r(1.0, 2.0), r(3.0, 4.0))
        val s = small.toString()
        assertTrue(s.startsWith("DenseMatrix(2 x 2)"), s)
        assertTrue(s.contains("1.0 2.0") && s.contains("3.0 4.0"), s)
        assertFalse(s.contains("…"), s)
        val big = DenseMatrix.zeros(8, 9).toString()
        assertTrue(big.contains("DenseMatrix(8 x 9)") && big.contains(" …") && big.endsWith("\n…"), big)
        assertEquals(1 + 6 + 1, big.lines().size, "6 preview rows + header + “…”")
        assertEquals("DenseMatrix(0 x 0)", DenseMatrix.zeros(0, 0).toString())
    }

    @Test
    fun `DenseMatrix equals and hashCode depend on shape and contents`() {
        val a = m(r(1.0, 2.0, 3.0), r(4.0, 5.0, 6.0))
        val b = m(r(1.0, 2.0, 3.0), r(4.0, 5.0, 6.0))
        val t = a.transpose()
        assertEquals(a, a)
        assertEquals(a, b); assertEquals(b, a)
        assertEquals(a.hashCode(), b.hashCode())
        assertNotEquals(a, t, "same data, different shape")
        assertNotEquals(a, m(r(1.0, 2.0, 3.0), r(4.0, 5.0, 7.0)))
        assertFalse(a.equals(null)); assertFalse(a.equals("DenseMatrix"))
        assertNotEquals(DenseMatrix.zeros(2, 3).hashCode(), DenseMatrix.zeros(3, 2).hashCode())
    }

    @Test
    fun `DenseMatrix row and column return copies and validate indices`() {
        val a = m(r(1.0, 2.0, 3.0), r(4.0, 5.0, 6.0))
        assertContentEquals(r(4.0, 5.0, 6.0), a.row(1))
        assertContentEquals(r(2.0, 5.0), a.column(1))
        a.row(0)[0] = 99.0; a.column(0)[0] = 99.0
        assertEquals(1.0, a[0, 0], "row/column must not share memory with the matrix")
        for (bad in listOf(-1, 2)) {
            val e = assertFailsWith<IllegalArgumentException> { a.row(bad) }
            assertTrue(e.message!!.contains("Row index"), e.message)
        }
        for (bad in listOf(-1, 3)) {
            val e = assertFailsWith<IllegalArgumentException> { a.column(bad) }
            assertTrue(e.message!!.contains("Column index"), e.message)
        }
    }

    @Test
    fun `DenseMatrix diagonal and fromColumnMajor with wrong length`() {
        val d = DenseMatrix.diagonal(r(1.0, 2.0, 3.0))
        assertEquals(3, d.rows); assertEquals(3, d.cols)
        for (i in 0 until 3) for (j in 0 until 3) assertEquals(if (i == j) (i + 1).toDouble() else 0.0, d[i, j])
        val e = assertFailsWith<IllegalArgumentException> { DenseMatrix.fromColumnMajor(2, 3, DoubleArray(5)) }
        assertTrue(e.message!!.contains("Array length") && e.message!!.contains("6"), e.message)
        val n = assertFailsWith<IllegalArgumentException> { DenseMatrix.fromColumnMajor(-1, 0, DoubleArray(0)) }
        assertTrue(n.message!!.contains("non-negative"), n.message)
    }

    @Test
    fun `Shapes messages for empty, non-square, mismatched and non-finite matrices`() {
        val empty = DenseMatrix.zeros(0, 3)
        assertTrue(assertFailsWith<IllegalArgumentException> { Shapes.requireNonEmpty(empty) }.message!!.contains("must not be empty"))
        assertTrue(assertFailsWith<IllegalArgumentException> { Shapes.requireSquare(empty, "A") }.message!!.contains("must not be empty"))
        val same = assertFailsWith<IllegalArgumentException> {
            Shapes.requireSameShape(DenseMatrix.zeros(2, 3), DenseMatrix.zeros(3, 2))
        }
        assertTrue(same.message!!.contains("must match") && same.message!!.contains("2×3"), same.message)
        val nan = m(r(1.0, Double.NaN), r(0.0, 1.0))
        assertTrue(assertFailsWith<IllegalArgumentException> { Shapes.requireFinite(nan, "A") }.message!!.contains("non-finite"))
        assertTrue(assertFailsWith<IllegalArgumentException> { Shapes.requireFinite(r(Double.POSITIVE_INFINITY), "b") }.message!!.contains("non-finite"))
        Shapes.requireFinite(DenseMatrix.identity(2), "A"); Shapes.requireFinite(r(1.0), "b")
        assertTrue(assertFailsWith<IllegalArgumentException> { Shapes.requireSymmetric(empty, 1.0) }.message!!.contains("must not be empty"))
        assertTrue(assertFailsWith<IllegalArgumentException> { Conditioning.inverse(DenseMatrix.zeros(0, 0)) }.message!!.contains("must not be empty"))
    }

    @Test
    fun `Backends resolve available describe`() {
        assertFalse(Backends.resolve("java").isNative)
        assertFalse(Backends.resolve(" JAVA ").isNative)
        if (Backends.isNativeAvailable()) {
            assertTrue(Backends.resolve("native").isNative)
        } else {
            assertFailsWith<IllegalStateException> { Backends.resolve("native") }
        }
        assertEquals(Backends.resolve(null).name, Backends.resolve("auto").name)
        assertEquals(Backends.resolve("").name, Backends.resolve("auto").name)
        val e = assertFailsWith<IllegalArgumentException> { Backends.resolve("bogus") }
        assertTrue(e.message!!.contains("bogus") && e.message!!.contains("native, java, auto"), e.message)
        val av = Backends.available()
        assertEquals(if (Backends.isNativeAvailable()) 2 else 1, av.size)
        assertFalse(av.last().isNative)
        if (!Backends.isNativeAvailable()) assertEquals(Backends.java(), av.single())
        assertTrue(Backends.describe().contains("numerics.backend="), Backends.describe())
        assertEquals(Backends.describe(), Backends.describe())
    }

    @Test
    fun `NetlibBackend Frobenius and max norms and LU on a singular matrix`() {
        val a = m(r(1.0, -2.0), r(3.0, 4.0))
        for (b in Backends.available()) {
            assertEquals(sqrt(1.0 + 4.0 + 9.0 + 16.0), b.norm(a, MatrixNorm.FROBENIUS), 1e-15, b.name)
            assertEquals(4.0, b.norm(a, MatrixNorm.MAX), b.name)
            assertEquals(6.0, b.norm(a, MatrixNorm.ONE), b.name)
            assertEquals(7.0, b.norm(a, MatrixNorm.INF), b.name)
            assertEquals(0.0, b.norm(DenseMatrix.zeros(0, 2), MatrixNorm.FROBENIUS), b.name)
            val lu = b.luFactor(m(r(1.0, 2.0), r(2.0, 4.0)))
            assertTrue(lu.isSingular, b.name)
            val e = assertFailsWith<IllegalArgumentException> { b.luSolve(lu, DenseMatrix.identity(2)) }
            assertTrue(e.message!!.contains("singular"), e.message)
            assertNull(b.inverse(m(r(1.0, 2.0), r(2.0, 4.0))), b.name)
            val y = r(1.0, 1.0); b.axpy(2.0, r(1.0, -1.0), y); assertContentEquals(r(3.0, -1.0), y)
            b.axpy(2.0, DoubleArray(0), DoubleArray(0))
            assertTrue(assertFailsWith<IllegalArgumentException> { b.axpy(1.0, r(1.0), r(1.0, 2.0)) }.message!!.contains("vector lengths must match"))
            assertEquals(0, b.matVec(DenseMatrix.zeros(0, 0), DoubleArray(0)).size)
            assertEquals(0, b.matTransVec(DenseMatrix.zeros(3, 0), r(0.0, 0.0, 0.0)).size)
            assertEquals(DenseMatrix.zeros(2, 0), b.matMat(DenseMatrix.zeros(2, 3), DenseMatrix.zeros(3, 0)))
            assertEquals(DenseMatrix.zeros(0, 2), b.matTransMat(DenseMatrix.zeros(3, 0), DenseMatrix.zeros(3, 2)))
            assertEquals(0, b.symmetricEigenvalues(DenseMatrix.zeros(0, 0)).size)
            assertEquals(1.0, b.reciprocalCondition1(LuFactorization(DenseMatrix.zeros(0, 0), IntArray(0), 0), 0.0))
        }
        assertTrue(LuFactorization(DenseMatrix.identity(2), intArrayOf(1, 2), 2).isSingular)
        assertFalse(LuFactorization(DenseMatrix.identity(2), intArrayOf(1, 2), 0).isSingular)
    }
}
