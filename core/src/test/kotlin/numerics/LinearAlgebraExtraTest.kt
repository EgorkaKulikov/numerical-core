package numerics

import org.junit.jupiter.api.Tag
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Additional LinearAlgebra tests: cover the zero-skipping branches in matMat/atWa,
 * transposed/vector operations, norms and row pivoting in LU solve.
 * Expected values are derived analytically inside the test — no magic numbers.
 */
@Tag("fast")
class LinearAlgebraExtraTest {
    private val tol = 1e-12

    /** zeros creates a zero matrix of the requested size. */
    @Test fun zerosShapeAndValues() {
        val z = LinearAlgebra.zeros(2, 3)
        assertEquals(2, z.size); assertEquals(3, z[0].size)
        for (row in z) for (v in row) assertEquals(0.0, v, tol)
    }

    /** matVec: A x for a 2x2 A and x — against a hand-computed row-by-vector product. */
    @Test fun matVecAgainstManual() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val x = doubleArrayOf(5.0, 6.0)
        val r = LinearAlgebra.matVec(a, x)
        assertEquals(1.0 * 5 + 2.0 * 6, r[0], tol)
        assertEquals(3.0 * 5 + 4.0 * 6, r[1], tol)
    }

    /** matTransVec: A^T y for a 2x2 A — the sum of columns weighted by y. */
    @Test fun matTransVecAgainstManual() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val y = doubleArrayOf(1.0, 1.0)
        val r = LinearAlgebra.matTransVec(a, y)
        assertEquals(1.0 + 3.0, r[0], tol) // first column
        assertEquals(2.0 + 4.0, r[1], tol) // second column
    }

    /** matMat: zero-element skipping branch (ail==0.0) when A contains zeros. */
    @Test fun matMatSkipsZeroEntries() {
        val a = arrayOf(doubleArrayOf(0.0, 1.0)) // first factor is zero -> continue
        val b = arrayOf(doubleArrayOf(5.0, 5.0), doubleArrayOf(2.0, 3.0))
        val r = LinearAlgebra.matMat(a, b)
        // the result equals the second row of B since the contribution of the zero element is skipped
        assertEquals(2.0, r[0][0], tol)
        assertEquals(3.0, r[0][1], tol)
    }

    /** atWa: branch skipping a zero rwi when the row weight is zero. */
    @Test fun atWaSkipsZeroWeight() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val w = doubleArrayOf(0.0, 1.0) // the first row has zero weight and is skipped
        val g = LinearAlgebra.atWa(a, w)
        // only the second row contributes: [3,4]^T [3,4]
        assertEquals(9.0, g[0][0], tol)
        assertEquals(12.0, g[0][1], tol)
        assertEquals(16.0, g[1][1], tol)
        assertTrue(LinearAlgebra.maxAsymmetry(g) < tol)
    }

    /** addScaled: A + s*B elementwise. */
    @Test fun addScaledElementwise() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0))
        val b = arrayOf(doubleArrayOf(3.0, 4.0))
        val r = LinearAlgebra.addScaled(a, b, 2.0)
        assertEquals(1.0 + 2 * 3.0, r[0][0], tol)
        assertEquals(2.0 + 2 * 4.0, r[0][1], tol)
        // input is unchanged
        assertEquals(1.0, a[0][0], tol)
    }

    /** norm2: Euclidean norm of (3,4) = 5. */
    @Test fun norm2Pythagorean() {
        assertEquals(5.0, LinearAlgebra.norm2(doubleArrayOf(3.0, 4.0)), tol)
        assertEquals(sqrt(2.0), LinearAlgebra.norm2(doubleArrayOf(1.0, 1.0)), tol)
    }

    /** normInf: maximum absolute coordinate. */
    @Test fun normInfMaxAbs() {
        assertEquals(7.0, LinearAlgebra.normInf(doubleArrayOf(-3.0, 2.0, -7.0)), tol)
    }

    /** maxAsymmetry: for a non-symmetric matrix equals max|A-A^T|. */
    @Test fun maxAsymmetryValue() {
        val a = arrayOf(doubleArrayOf(0.0, 5.0), doubleArrayOf(2.0, 0.0))
        assertEquals(3.0, LinearAlgebra.maxAsymmetry(a), tol) // |5-2|
    }

    /** solve: row-pivoting branch (zero pivot in column 0). */
    @Test fun solveWithRowSwap() {
        val a = arrayOf(doubleArrayOf(0.0, 2.0), doubleArrayOf(1.0, 1.0))
        val b = doubleArrayOf(2.0, 3.0)
        val x = LinearAlgebra.solve(a, b)
        // check by substitution: 0*x0+2*x1=2 -> x1=1; x0+x1=3 -> x0=2
        assertEquals(2.0, x[0], 1e-10)
        assertEquals(1.0, x[1], 1e-10)
    }

    /** solve: back substitution on an upper-triangular 3x3 system. */
    @Test fun solveBackSubstitution() {
        val a = arrayOf(
            doubleArrayOf(2.0, 1.0, 1.0),
            doubleArrayOf(0.0, 3.0, 1.0),
            doubleArrayOf(0.0, 0.0, 4.0),
        )
        val b = doubleArrayOf(2.0 + 2.0 + 3.0, 6.0 + 3.0, 12.0) // solution (1,2,3)
        val x = LinearAlgebra.solve(a, b)
        assertEquals(1.0, x[0], 1e-10)
        assertEquals(2.0, x[1], 1e-10)
        assertEquals(3.0, x[2], 1e-10)
    }

    /** Regression #6: empty/inconsistent input to public methods -> a clear exception. */
    @Test fun emptyAndMismatchedInputsThrow() {
        val empty = emptyArray<DoubleArray>()
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        // matVec
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.matVec(empty, doubleArrayOf()) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.matVec(a, doubleArrayOf(1.0)) }
        // matMat
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.matMat(empty, a) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.matMat(a, arrayOf(doubleArrayOf(1.0))) }
        // atWa
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.atWa(empty, doubleArrayOf()) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.atWa(a, doubleArrayOf(1.0)) }
        // solve
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.solve(empty, doubleArrayOf()) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.solve(a, doubleArrayOf(1.0)) }
    }

    /**
     * solve requires a square A: the backends assume n x n, and without the check a rectangular
     * input would yield either a deep internal exception or a silently wrong answer.
     */
    @Test fun solveRejectsNonSquareMatrix() {
        // 2x3: fewer rows than columns
        val wide = arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0))
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.solve(wide, doubleArrayOf(1.0, 2.0)) }
        // 3x2: fewer columns than rows
        val tall = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0), doubleArrayOf(5.0, 6.0))
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.solve(tall, doubleArrayOf(1.0, 2.0, 3.0)) }
    }

    /** solve rejects a ragged matrix (first row has the right length, the others do not). */
    @Test fun solveRejectsRaggedMatrix() {
        val ragged = arrayOf(
            doubleArrayOf(1.0, 0.0, 0.0),
            doubleArrayOf(0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0),
        )
        val e = assertFailsWith<IllegalArgumentException> {
            LinearAlgebra.solve(ragged, doubleArrayOf(1.0, 1.0, 1.0))
        }
        assertTrue(e.message!!.contains("1"), "message must name the row index: ${e.message}")
    }

    /** maxAsymmetry is defined only for a square A: a[i][j] is indexed over a.indices. */
    @Test fun maxAsymmetryRejectsNonSquareAndRagged() {
        val wide = arrayOf(doubleArrayOf(1.0, 2.0, 3.0), doubleArrayOf(4.0, 5.0, 6.0))
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.maxAsymmetry(wide) }
        val ragged = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0))
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.maxAsymmetry(ragged) }
        assertFailsWith<IllegalArgumentException> { LinearAlgebra.maxAsymmetry(emptyArray()) }
    }

    /** Regression #6: valid operations keep working after the guard was added. */
    @Test fun validOperationsStillWork() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0))
        val r = LinearAlgebra.matVec(a, doubleArrayOf(1.0, 1.0))
        assertEquals(3.0, r[0], tol)
        assertEquals(7.0, r[1], tol)
    }
}
