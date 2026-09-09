package numerics

import java.util.Random
import kotlin.math.abs
import kotlin.math.sin
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertSame
import kotlin.test.assertTrue
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestFactory

/**
 * Контракт [DenseMatrix] и согласие перегрузок фасада на [DenseMatrix] с перегрузками
 * над [Array]<[DoubleArray]>: адаптеры обязаны давать побитово тот же результат.
 */
@Tag("fast")
class DenseMatrixTest {

    private fun randomRows(rng: Random, rows: Int, cols: Int): Array<DoubleArray> =
        Array(rows) { DoubleArray(cols) { rng.nextDouble() * 2.0 - 1.0 } }

    /** Симметричная положительно определённая A^T A + n·I. */
    private fun spdRows(rng: Random, n: Int): Array<DoubleArray> {
        val a = randomRows(rng, n, n)
        return Array(n) { i ->
            DoubleArray(n) { j ->
                var s = if (i == j) n.toDouble() else 0.0
                for (k in 0 until n) s += a[k][i] * a[k][j]
                s
            }
        }
    }

    private fun assertRowsBitwise(expected: Array<DoubleArray>, actual: DenseMatrix, what: String) {
        assertEquals(expected.size, actual.rows, "$what: число строк")
        for (i in expected.indices) assertContentEquals(expected[i], actual.row(i), "$what: строка $i")
    }

    // --- конструкторы и валидация --------------------------------------------

    @Test
    fun `отрицательные размеры отвергаются`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix.zeros(-1, 3) }
        assertFailsWith<IllegalArgumentException> { DenseMatrix.zeros(3, -1) }
        assertFailsWith<IllegalArgumentException> { DenseMatrix.fromColumnMajor(-2, 2, DoubleArray(4)) }
        assertFailsWith<IllegalArgumentException> { DenseMatrix.build(-1, 1) { _, _ -> 0.0 } }
        assertFailsWith<IllegalArgumentException> { ParallelAssembly.assembleDense(-1, 1) { _, _ -> 0.0 } }
    }

    @Test
    fun `несогласованная длина массива отвергается`() {
        assertFailsWith<IllegalArgumentException> { DenseMatrix.fromColumnMajor(2, 2, DoubleArray(3)) }
        assertFailsWith<IllegalArgumentException> { DenseMatrix.fromColumnMajor(2, 2, DoubleArray(5)) }
    }

    @Test
    fun `рваный fromRows отвергается с указанием строки`() {
        val ragged = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(3.0), doubleArrayOf(4.0, 5.0))
        val e = assertFailsWith<IllegalArgumentException> { DenseMatrix.fromRows(ragged) }
        assertTrue(e.message!!.contains("Строка 1"), e.message)
    }

    @TestFactory
    fun `пустые матрицы`(): List<DynamicTest> = listOf(
        DynamicTest.dynamicTest("0x0 из пустого массива") {
            val m = DenseMatrix.fromRows(emptyArray())
            assertEquals(0, m.rows)
            assertEquals(0, m.cols)
            assertEquals(0, m.data.size)
            assertEquals(0, m.toRows().size)
            assertEquals(m, m.transpose())
        },
        DynamicTest.dynamicTest("0x5") {
            val m = DenseMatrix.zeros(0, 5)
            assertEquals(0, m.rows)
            assertEquals(5, m.cols)
            assertEquals(0, m.data.size)
            assertEquals(0, m.toRows().size)
            val t = m.transpose()
            assertEquals(5, t.rows)
            assertEquals(0, t.cols)
        },
        DynamicTest.dynamicTest("5x0") {
            val m = DenseMatrix.zeros(5, 0)
            assertEquals(5, m.rows)
            assertEquals(0, m.cols)
            val rows = m.toRows()
            assertEquals(5, rows.size)
            assertEquals(0, rows[0].size)
            assertEquals(m, DenseMatrix.fromRows(rows))
        },
    )

    @Test
    fun `identity diagonal build и столбцовая раскладка`() {
        val id = DenseMatrix.identity(3)
        assertEquals(DenseMatrix.diagonal(doubleArrayOf(1.0, 1.0, 1.0)), id)
        assertEquals(LinearAlgebra.identityMatrix(3), id)
        assertEquals(LinearAlgebra.zerosMatrix(2, 3), DenseMatrix.zeros(2, 3))
        val m = DenseMatrix.build(2, 3) { i, j -> (10 * i + j).toDouble() }
        assertContentEquals(doubleArrayOf(0.0, 10.0, 1.0, 11.0, 2.0, 12.0), m.data)
        assertContentEquals(doubleArrayOf(10.0, 11.0, 12.0), m.row(1))
        assertContentEquals(doubleArrayOf(2.0, 12.0), m.column(2))
        assertEquals(11.0, m[1, 1])
        assertTrue(id.isSquare)
        assertTrue(!m.isSquare)
    }

    // --- get/set -------------------------------------------------------------

    @TestFactory
    fun `выход за границы в get и set`(): List<DynamicTest> {
        val m = DenseMatrix.zeros(2, 3)
        return listOf(-1 to 0, 2 to 0, 0 to -1, 0 to 3, 2 to 3).map { (i, j) ->
            DynamicTest.dynamicTest("($i, $j) в 2x3") {
                assertFailsWith<IllegalArgumentException> { m[i, j] }
                assertFailsWith<IllegalArgumentException> { m[i, j] = 1.0 }
            }
        }
    }

    @Test
    fun `set пишет в data а copy независим`() {
        val m = DenseMatrix.zeros(3, 2)
        m[2, 1] = 7.0
        assertEquals(7.0, m.data[2 + 1 * 3])
        val c = m.copy()
        c[0, 0] = 1.0
        assertEquals(0.0, m[0, 0])
        assertEquals(1.0, c[0, 0])
    }

    // --- перекладка формата --------------------------------------------------

    private val shapes = listOf(1 to 1, 2 to 3, 3 to 2, 5 to 5, 7 to 1, 1 to 7, 16 to 9)

    @TestFactory
    fun `fromRows после toRows побитово возвращает матрицу`(): List<DynamicTest> = shapes.map { (r, c) ->
        DynamicTest.dynamicTest("${r}x$c") {
            val rows = randomRows(Random(1000L + r * 31 + c), r, c)
            val m = DenseMatrix.fromRows(rows)
            assertRowsBitwise(rows, m, "fromRows")
            assertEquals(m, DenseMatrix.fromRows(m.toRows()))
        }
    }

    @TestFactory
    fun `двойное транспонирование тождественно`(): List<DynamicTest> = shapes.map { (r, c) ->
        DynamicTest.dynamicTest("${r}x$c") {
            val m = DenseMatrix.fromRows(randomRows(Random(2000L + r * 31 + c), r, c))
            val t = m.transpose()
            assertEquals(c, t.rows)
            assertEquals(r, t.cols)
            for (i in 0 until r) for (j in 0 until c) assertEquals(m[i, j], t[j, i])
            assertEquals(m, t.transpose())
        }
    }

    @Test
    fun `fromColumnMajor не копирует массив`() {
        val arr = DoubleArray(6)
        val m = DenseMatrix.fromColumnMajor(2, 3, arr)
        assertSame(arr, m.data)
        arr[3] = 5.0
        assertEquals(5.0, m[1, 1])
        m[0, 2] = 9.0
        assertEquals(9.0, arr[4])
    }

    // --- согласие с фасадом --------------------------------------------------

    private val sizes = listOf(1, 2, 3, 5, 8, 16)

    @TestFactory
    fun `identity умножить A равно A через matMat`(): List<DynamicTest> = sizes.map { n ->
        DynamicTest.dynamicTest("n=$n") {
            val a = DenseMatrix.fromRows(randomRows(Random(3000L + n), n, n))
            val p = LinearAlgebra.matMat(LinearAlgebra.identityMatrix(n), a)
            assertEquals(n, p.rows)
            assertEquals(n, p.cols)
            for (i in 0 until n) for (j in 0 until n) {
                assertTrue(abs(p[i, j] - a[i, j]) <= 1e-15, "($i,$j): ${p[i, j]} vs ${a[i, j]}")
            }
        }
    }

    @Suppress("DEPRECATION")
    @TestFactory
    fun `перегрузки DenseMatrix побитово совпадают с Array`(): List<DynamicTest> = sizes.flatMap { n ->
        val rng = Random(4000L + n)
        val rect = randomRows(rng, n, n + 1)
        val rect2 = randomRows(rng, n, n + 1)
        val tall = randomRows(rng, n + 1, n)
        val x = DoubleArray(n + 1) { rng.nextDouble() }
        val y = DoubleArray(n) { rng.nextDouble() }
        val square = randomRows(rng, n, n).also { a -> for (i in 0 until n) a[i][i] += n + 1.0 }
        val spd = spdRows(rng, n)
        val dRect = DenseMatrix.fromRows(rect)
        val dRect2 = DenseMatrix.fromRows(rect2)
        val dTall = DenseMatrix.fromRows(tall)
        val dSquare = DenseMatrix.fromRows(square)
        val dSpd = DenseMatrix.fromRows(spd)
        val cases = listOf<Pair<String, () -> Unit>>(
            "matVec" to { assertContentEquals(LinearAlgebra.matVec(rect, x), LinearAlgebra.matVec(dRect, x)) },
            "matTransVec" to { assertContentEquals(LinearAlgebra.matTransVec(rect, y), LinearAlgebra.matTransVec(dRect, y)) },
            "matMat" to { assertRowsBitwise(LinearAlgebra.matMat(rect, tall), LinearAlgebra.matMat(dRect, dTall), "matMat") },
            "atWa" to { assertRowsBitwise(LinearAlgebra.atWa(rect, y), LinearAlgebra.atWa(dRect, y), "atWa") },
            "addScaled" to {
                assertRowsBitwise(LinearAlgebra.addScaled(rect, rect2, -0.7), LinearAlgebra.addScaled(dRect, dRect2, -0.7), "addScaled")
            },
            "solve" to { assertContentEquals(LinearAlgebra.solve(square, y), LinearAlgebra.solve(dSquare, y)) },
            "solveDiagnosed" to {
                assertContentEquals(LinearAlgebra.solveDiagnosed(square, y).x, LinearAlgebra.solveDiagnosed(dSquare, y).x)
            },
            "cholesky" to {
                val expected = assertNotNull(LinearAlgebra.cholesky(spd))
                assertRowsBitwise(expected, assertNotNull(LinearAlgebra.cholesky(dSpd)), "cholesky")
            },
            "maxAsymmetry" to { assertEquals(LinearAlgebra.maxAsymmetry(square), LinearAlgebra.maxAsymmetry(dSquare)) },
        )
        cases.map { (name, body) -> DynamicTest.dynamicTest("$name, n=$n") { body() } }
    }

    // --- параллельная сборка -------------------------------------------------

    @TestFactory
    fun `assembleDense побитово равен assembleMatrix`(): List<DynamicTest> =
        listOf(1 to 1, 3 to 5, 17 to 4, 200 to 200).flatMap { (r, c) ->
            listOf(true, false).map { parallel ->
                DynamicTest.dynamicTest("${r}x$c, parallel=$parallel") {
                    val cell = { i: Int, j: Int -> sin(0.37 * i + 1.13 * j) / (1.0 + i + j) }
                    val expected = DenseMatrix.fromRows(ParallelAssembly.assembleMatrix(r, c, parallel, cell))
                    val actual = ParallelAssembly.assembleDense(r, c, parallel, cell)
                    assertEquals(expected, actual)
                }
            }
        }
}
