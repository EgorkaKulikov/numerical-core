package numerics.backend

import numerics.LinearAlgebra
import numerics.NumericsContext
import org.junit.jupiter.api.Tag
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Тесты подключаемого SPI линейной алгебры [LinAlgBackend]/[Backends].
 *
 * Проверяют реальную доступность бэкендов, выбор по умолчанию ([Backends.default]),
 * согласие двух бэкендов на хорошо обусловленных входах, делегирование фасада
 * ЯВНО переданному бэкенду и сохранение семантики вырожденности.
 *
 * Ни один тест ничего НЕ ВОССТАНАВЛИВАЕТ в `finally` и не может «протечь» в соседний:
 * глобального мутируемого состояния больше нет — бэкенд передаётся параметром,
 * и восстанавливать нечего. Именно ради этого свойства убирались `Backends.use/reset`.
 */
@Tag("fast")
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

    /**
     * На этой машине нативный multik/OpenBLAS доступен — и именно он выбирается
     * автоматически (без явного `-Dnumerics.backend`).
     *
     * Смысл теста сохранён, но он больше НЕ требует `Backends.default() == MultikCpuBackend`:
     * фаст-набор обязан прогоняться НА ОБОИХ бэкендах (в том числе с
     * `-Dnumerics.backend=reference`), а жёсткая проверка активного бэкенда это
     * блокировала. Проверяется два факта: (1) нативная библиотека реально грузится;
     * (2) авто-выбор (requested = null) предпочитает его reference-бэкенду.
     */
    @Test
    fun multikCpuBackendIsAvailableHere() {
        assertTrue(MultikCpuBackend.isAvailable(), "MultikCpuBackend должен быть доступен на этой машине")
        assertEquals(
            MultikCpuBackend,
            Backends.select(requested = null),
            "При авто-выборе на этой машине должен выигрывать multik CPU бэкенд"
        )
    }

    /**
     * Бэкенд по умолчанию соответствует запрошенному через `-Dnumerics.backend`
     * (а при отсутствии свойства — результату авто-выбора).
     *
     * Именно этот тест ловит молчаливую подмену бэкенда в реальном прогоне.
     */
    @Test
    fun defaultBackendMatchesRequestedProperty() {
        val requested = System.getProperty("numerics.backend")?.trim()?.lowercase()
        // НЕИЗВЕСТНОЕ имя — НЕ то же самое, что «свойство не задано»: по контракту
        // [Backends.default] обязан бросить, а не вернуть авто-выбор. Сравнение через
        // assertEquals в этой ветке ложно диагностировало бы КОРРЕКТНОЕ поведение
        // как падение (прогон с -Dnumerics.backend=nosuchbackend).
        val known = requested == null || requested.isEmpty() ||
            requested == "multik" || requested == "reference"
        if (!known) {
            val ex = assertFailsWith<IllegalStateException> { Backends.default() }
            assertTrue(
                ex.message!!.contains("неизвестный бэкенд"),
                "Сообщение должно называть причину: ${ex.message}"
            )
            return
        }
        val expected = when (requested) {
            "multik" -> MultikCpuBackend
            "reference" -> ReferenceBackend
            else -> Backends.select(requested = null)
        }
        assertEquals(expected, Backends.default(), "Бэкенд по умолчанию расходится с запрошенным")
    }

    /**
     * Д1: ПРИ ЯВНО ЗАПРОШЕННОМ и НЕДОСТУПНОМ бэкенде — ошибка, а НЕ молчаливый
     * откат на reference: подмена бэкенда меняет численные результаты (на задачах
     * первого рода — до процентов), и пользователь должен узнать об этом сразу.
     */
    @Test
    fun explicitlyRequestedUnavailableBackendFailsLoudly() {
        val ex = assertFailsWith<IllegalStateException> {
            Backends.select(requested = "multik", isAvailable = { it !== MultikCpuBackend })
        }
        assertTrue(
            ex.message!!.contains("недоступен"),
            "Сообщение должно объяснять недоступность: ${ex.message}"
        )
    }

    /** Неизвестное имя бэкенда тоже не должно игнорироваться молча. */
    @Test
    fun unknownRequestedBackendFailsLoudly() {
        assertFailsWith<IllegalStateException> { Backends.select(requested = "cuda") }
    }

    /** Откат ПРИ ОТСУТСТВИИ свойства сохранён: там он — гарантия переносимости. */
    @Test
    fun autoSelectionStillFallsBackWhenNothingRequested() {
        assertEquals(
            ReferenceBackend,
            Backends.select(requested = null, isAvailable = { it !== MultikCpuBackend }),
        )
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

    /**
     * Фасад LinearAlgebra считает ТЕМ бэкендом, который ему передали ЯВНО.
     *
     * Раньше здесь подменялось глобальное состояние (`Backends.use`) с восстановлением
     * в finally, и корректность соседних тестов зависела от того, выполнился ли finally.
     * Теперь бэкенд — параметр, и восстанавливать нечего.
     */
    @Test
    fun facadeUsesExplicitlyPassedBackend() {
        // Известная система: [[2,0],[0,4]] x = [2,8] -> x = [1,2].
        val a = arrayOf(doubleArrayOf(2.0, 0.0), doubleArrayOf(0.0, 4.0))
        val b = doubleArrayOf(2.0, 8.0)
        val expected = doubleArrayOf(1.0, 2.0)
        for (backend in listOf<LinAlgBackend>(ReferenceBackend, MultikCpuBackend)) {
            assertVecEq(expected, LinearAlgebra.solve(a, b, backend))
            assertVecEq(expected, LinearAlgebra.matVec(LinearAlgebra.identity(2), expected, backend))
        }
        // Контекст по умолчанию несёт ровно стартовый бэкенд процесса.
        assertEquals(Backends.default(), NumericsContext.default().backend)
        assertTrue(NumericsContext.default().parallel, "По умолчанию сборка параллельная")
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
