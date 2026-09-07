package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * REGRESSION-ТЕСТЫ обнаруженных и исправленных дефектов линейной алгебры.
 *
 * Каждый тест назван по дефекту и снабжён описанием: в чём была ошибка, почему она
 * не проявлялась в существующих тестах и что именно проверяется теперь. Все эти
 * тесты ПАДАЮТ на коде до исправления. Нумерация дефектов общая с репозиторием
 * `integral-equations` (`regression.DefectRegressionTest`), откуда эти два теста
 * выделены как относящиеся к слою линейной алгебры.
 */
@Tag("fast")
class LinearAlgebraRegressionTest {

    /**
     * ДЕФЕКТ 4. Порог вырожденности в эталонной линейной алгебре был абсолютным
     * (1e-300) и фактически проверял лишь строгий машинный ноль: практически
     * вырожденная матрица решалась молча и возвращала бессмысленный результат.
     *
     * Теперь порог относителен норме матрицы, поэтому вырожденность распознаётся
     * независимо от масштаба данных.
     */
    @Test
    fun defect4_singularityDetectedRegardlessOfScale() {
        for (scale in listOf(1.0, 1e6, 1e-6)) {
            val singular = arrayOf(
                doubleArrayOf(1.0 * scale, 2.0 * scale),
                doubleArrayOf(2.0 * scale, 4.0 * scale),
            )
            assertFailsWith<IllegalStateException>("Масштаб $scale: вырожденность должна распознаваться") {
                ReferenceLinearAlgebra.solve(singular, doubleArrayOf(1.0 * scale, 2.0 * scale))
            }
        }
        for (scale in listOf(1.0, 1e6, 1e-6)) {
            val regular = arrayOf(
                doubleArrayOf(1.0 * scale, 2.0 * scale),
                doubleArrayOf(3.0 * scale, 4.0 * scale),
            )
            val solution = ReferenceLinearAlgebra.solve(regular, doubleArrayOf(1.0 * scale, 1.0 * scale))
            assertTrue(solution.all { it.isFinite() }, "Масштаб $scale: решение должно быть конечным")
        }
    }

    /**
     * ДЕФЕКТ 5. Бэкенды линейной алгебры расходились на нечисловом входе:
     * multik/OpenBLAS бросал `IllegalStateException`, а эталонная реализация молча
     * возвращала вектор из NaN. Наблюдаемое поведение обязано совпадать.
     */
    @Test
    fun defect5_backendsAgreeOnNonFiniteInput() {
        val withNaN = arrayOf(
            doubleArrayOf(Double.NaN, 1.0),
            doubleArrayOf(1.0, 1.0),
        )
        assertFailsWith<IllegalStateException>(
            "Эталонная реализация обязана сигнализировать об ошибке так же, как нативный бэкенд",
        ) {
            ReferenceLinearAlgebra.solve(withNaN, doubleArrayOf(1.0, 1.0))
        }
    }
}
