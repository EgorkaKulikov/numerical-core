package numerics.backend

import numerics.LinearAlgebra
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Тесты подключаемого SPI линейной алгебры [LinAlgBackend]/[Backends].
 *
 * Проверяют реальную доступность бэкендов, выбор по умолчанию, согласие двух
 * бэкендов на хорошо обусловленных входах, делегирование фасада активному
 * бэкенду и сохранение семантики вырожденности. Каждый тест восстанавливает
 * бэкенд по умолчанию в finally, чтобы порядок тестов не мог «протечь».
 */
class BackendSpiTest {

    private val tol = 1e-9

    private fun randMatrix(rnd: Random, rows: Int, cols: Int): Array<DoubleArray> =
        Array(rows) { DoubleArray(cols) { rnd.nextDouble(-1.0, 1.0) } }

    private fun randVector(rnd: Random, n: Int): DoubleArray =
        DoubleArray(n) { rnd.nextDouble(-1.0, 1.0) }

    /** Диагонально доминирующая (невырожденная) матрица n x n. */
    private fun diagDominant(rnd: Random, n: Int): Array<DoubleArray> {
        val a = randMatrix(rnd, n, n)
        for (i in 0 until n) {
            var rowSum = 0.0
            for (j in 0 until n) rowSum += kotlin.math.abs(a[i][j])
            a[i][i] += rowSum + 1.0
        }
        return a
    }

    private fun assertMatEq(expected: Array<DoubleArray>, actual: Array<DoubleArray>) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            assertEquals(expected[i].size, actual[i].size)
            for (j in expected[i].indices) {
                assertTrue(
                    kotlin.math.abs(expected[i][j] - actual[i][j]) < tol,
                    "mismatch at [$i][$j]: ${expected[i][j]} vs ${actual[i][j]}"
                )
            }
        }
    }

    private fun assertVecEq(expected: DoubleArray, actual: DoubleArray) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) {
            assertTrue(
                kotlin.math.abs(expected[i] - actual[i]) < tol,
                "mismatch at [$i]: ${expected[i]} vs ${actual[i]}"
            )
        }
    }

    /** На этой машине нативный multik/OpenBLAS доступен и выбран по умолчанию. */
    @Test
    fun multikCpuBackendIsAvailableHere() {
        assertTrue(MultikCpuBackend.isAvailable(), "MultikCpuBackend должен быть доступен на этой машине")
        assertEquals(MultikCpuBackend, Backends.active, "По умолчанию активен multik CPU бэкенд")
    }

    /** Чистый JVM эталонный бэкенд доступен всегда. */
    @Test
    fun referenceBackendAlwaysAvailable() {
        assertTrue(ReferenceBackend.isAvailable())
    }

    /** На фиксированных хорошо обусловленных входах оба бэкенда согласны до 1e-9. */
    @Test
    fun bothBackendsAgree() {
        for (n in intArrayOf(3, 8, 20)) {
            val rnd = Random(7000 + n)
            val a = randMatrix(rnd, n, n)
            val bMat = randMatrix(rnd, n, n)
            val x = randVector(rnd, n)
            val w = randVector(rnd, n)
            val spd = diagDominant(rnd, n)
            val rhs = randVector(rnd, n)

            assertMatEq(MultikCpuBackend.matMat(a, bMat), ReferenceBackend.matMat(a, bMat))
            assertVecEq(MultikCpuBackend.matVec(a, x), ReferenceBackend.matVec(a, x))
            assertMatEq(MultikCpuBackend.atWa(a, w), ReferenceBackend.atWa(a, w))
            assertVecEq(MultikCpuBackend.solve(spd, rhs), ReferenceBackend.solve(spd, rhs))
        }
    }

    /** Фасад LinearAlgebra делегирует активному бэкенду, переключение работает. */
    @Test
    fun facadeFollowsActiveBackend() {
        try {
            Backends.use(ReferenceBackend)
            // Известная система: [[2,0],[0,4]] x = [2,8] -> x = [1,2].
            val a = arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 4.0))
            val b = doubleArrayOf(2.0, 8.0)
            val x = LinearAlgebra.solve(a, b)
            assertVecEq(doubleArrayOf(1.0, 2.0), x)
        } finally {
            Backends.use(MultikCpuBackend)
        }
    }

    /** Оба бэкенда бросают IllegalStateException на вырожденной матрице. */
    @Test
    fun solveThrowsOnSingularForBothBackends() {
        val a = arrayOf(doubleArrayOf(1.0, 2.0), doubleArrayOf(2.0, 4.0))
        val b = doubleArrayOf(1.0, 2.0)
        assertFailsWith<IllegalStateException> { MultikCpuBackend.solve(a, b) }
        assertFailsWith<IllegalStateException> { ReferenceBackend.solve(a, b) }
    }
}
