package numerics

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertNotNull
import kotlin.test.assertTrue
import org.junit.jupiter.api.assertThrows

class LinearAlgebraTest {
    private val tol = 1e-12

    @Test fun solve2x2() {
        // 2x + y = 5 ; x + 3y = 10  -> x=1, y=3
        val a = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))
        val b = doubleArrayOf(5.0, 10.0)
        val x = LinearAlgebra.solve(a, b)
        assertEquals(1.0, x[0], 1e-10)
        assertEquals(3.0, x[1], 1e-10)
    }

    @Test fun solve3x3() {
        // identity-shifted system with known solution
        val a = arrayOf(
            doubleArrayOf(2.0, 0.0, 0.0),
            doubleArrayOf(0.0, 4.0, 0.0),
            doubleArrayOf(0.0, 0.0, 5.0),
        )
        val b = doubleArrayOf(2.0, 8.0, 15.0)
        val x = LinearAlgebra.solve(a, b)
        assertEquals(1.0, x[0], 1e-10)
        assertEquals(2.0, x[1], 1e-10)
        assertEquals(3.0, x[2], 1e-10)
    }

    @Test fun solveDoesNotMutateInput() {
        val a = arrayOf(doubleArrayOf(2.0, 1.0), doubleArrayOf(1.0, 3.0))
        val b = doubleArrayOf(5.0, 10.0)
        LinearAlgebra.solve(a, b)
        assertEquals(2.0, a[0][0]); assertEquals(5.0, b[0])
    }

    @Test fun identityAndMatMat() {
        val id = LinearAlgebra.identity(3)
        val m = arrayOf(
            doubleArrayOf(1.0, 2.0, 3.0),
            doubleArrayOf(4.0, 5.0, 6.0),
            doubleArrayOf(7.0, 8.0, 9.0),
        )
        val prod = LinearAlgebra.matMat(id, m)
        for (i in 0..2) for (j in 0..2) assertEquals(m[i][j], prod[i][j], tol)
    }

    @Test fun atWaEqualsAtA_whenWeightsOne() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0, 4.0), doubleArrayOf(5.0, 6.0))
        val w = doubleArrayOf(1.0, 1.0, 1.0)
        val g = LinearAlgebra.atWa(a, w)
        // A^T A computed directly
        val at = arrayOf(doubleArrayOf(1.0, 3.0, 5.0), doubleArrayOf(2.0, 4.0, 6.0))
        val expected = LinearAlgebra.matMat(at, a)
        for (i in 0..1) for (j in 0..1) assertEquals(expected[i][j], g[i][j], 1e-9)
        // symmetry
        assertTrue(LinearAlgebra.maxAsymmetry(g) < 1e-12)
    }

    @Test fun choleskyOnSpd() {
        val a = arrayOf(
            doubleArrayOf(4.0, 2.0), doubleArrayOf(2.0, 3.0))
        val l = LinearAlgebra.cholesky(a)
        assertNotNull(l)
        // reconstruct L L^T
        for (i in 0..1) for (j in 0..1) {
            var s = 0.0
            for (k in 0..1) s += l!![i][k] * l[j][k]
            assertEquals(a[i][j], s, 1e-10)
        }
    }

    @Test fun choleskyNullOnNonSpd() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 1.0)) // indefinite
        assertNull(LinearAlgebra.cholesky(a))
    }

    @Test fun solveThrowsOnSingular() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0))
        val b = doubleArrayOf(1.0, 2.0)
        assertThrows<IllegalStateException> { LinearAlgebra.solve(a, b) }
    }
}
