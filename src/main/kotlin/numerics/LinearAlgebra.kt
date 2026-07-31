package numerics

import kotlin.math.abs
import numerics.backend.Backends

/**
 * Линейная алгебра над [Array]<[DoubleArray]> — тонкий фасад над подключаемым
 * бэкендом (SPI).
 *
 * Публичный API стабилен (массивы Kotlin) и не зависит от выбранной реализации.
 * Тяжёлые операции (умножения и решение СЛАУ) делегируются активному бэкенду
 * [numerics.backend.Backends.active]; по умолчанию это нативный
 * multik/OpenBLAS-бэкенд, с автоматическим откатом на чистый JVM при отсутствии
 * нативной библиотеки. Дешёвые скалярные/служебные операции (нормы, проверка
 * симметрии, разложение Холецкого как health-check, конструкторы) реализованы
 * прямо здесь. Эталоном корректности служит [ReferenceLinearAlgebra].
 *
 * Чтобы подключить новый бэкенд (например, GPU), реализуйте
 * [numerics.backend.LinAlgBackend] и зарегистрируйте его в
 * [numerics.backend.Backends] — этот фасад менять не нужно.
 */
object LinearAlgebra {

    // --- Тривиальные конструкторы (без бэкенда) ------------------------------

    /** Создаёт нулевую матрицу размера rows x cols. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = DenseOps.zeros(rows, cols)

    /** Единичная матрица размера n x n. */
    fun identity(n: Int): Array<DoubleArray> = DenseOps.identity(n)

    // --- Тяжёлые операции: делегирование активному бэкенду --------------------

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matVec: пустая матрица A" }
        require(a[0].size == x.size) { "matVec: несогласованные размеры A(${a.size}x${a[0].size}) и x(${x.size})" }
        return Backends.active.matVec(a, x)
    }

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matTransVec: пустая матрица A" }
        require(a.size == y.size) { "matTransVec: несогласованные размеры A(${a.size} строк) и y(${y.size})" }
        return Backends.active.matTransVec(a, y)
    }

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matMat: пустая матрица A" }
        require(b.isNotEmpty() && b[0].isNotEmpty()) { "matMat: пустая матрица B" }
        require(a[0].size == b.size) { "matMat: несогласованные размеры A(${a.size}x${a[0].size}) и B(${b.size}x${b[0].size})" }
        return Backends.active.matMat(a, b)
    }

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "atWa: пустая матрица A" }
        require(a.size == w.size) { "atWa: несогласованные размеры A(${a.size} строк) и w(${w.size})" }
        return Backends.active.atWa(a, w)
    }

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> {
        require(a.size == b.size) { "addScaled: несогласованное число строк A(${a.size}) и B(${b.size})" }
        require(a.isEmpty() || a[0].size == b[0].size) {
            "addScaled: несогласованное число столбцов A(${a[0].size}) и B(${b[0].size})"
        }
        return Backends.active.addScaled(a, b, s)
    }

    /**
     * Относительный допуск на НЕВЯЗКУ решения СЛАУ, общий для ВСЕХ бэкендов.
     *
     * Критерий: `||Ax-b||_inf <= SINGULARITY_RELATIVE_TOLERANCE * max(||A||_inf*||x||_inf, ||b||_inf)`.
     *
     * ПОЧЕМУ ИМЕННО НЕВЯЗКА, а не число обусловленности. Невязка — единственный
     * критерий, общий для любого бэкенда (нативный LAPACK не отдаёт ни ведущие
     * элементы, ни оценку cond) и при этом дешёвый: O(n^2) против O(n^3) самого
     * решения. Проверка по обусловленности здесь БЫЛА БЫ НЕВЕРНА по сути: задача
     * F1 (уравнение первого рода с регуляризацией alpha=1e-10) штатно даёт
     * `cond_2(I-M) ~ 1e10` при `||g||_inf ~ 1.6e10`, и отбраковывать её нельзя — это
     * штатный режим метода. Невязка же там ~1e-16 относительно масштаба, то есть
     * система решается обратно устойчиво, и проверка не срабатывает.
     *
     * ОТКУДА 1e-10. LU с частичным выбором обратно устойчив: гарантирует
     * `||Ax-b|| <= c*n*eps*||A||*||x||` с eps=2.2e-16 и умеренным фактором роста.
     * Для размеров задачи (n <= несколько сотен) это ~1e-13; запас в три порядка
     * оставлен сознательно: проверка обязана ловить МУСОР (невязка порядка
     * самого масштаба), а не различать оттенки качества у двух честных LU.
     *
     * Зачем `max(..., ||b||_inf)`: при почти нулевом x (например, b порядка
     * денормали) произведение `||A||*||x||` само вырождается в ноль, и любая
     * ненулевая ошибка округления дала бы ложное «вырождение».
     *
     * ГРАНИЦА ПРИМЕНИМОСТИ (важно для понимания контракта). Проверка гарантирует,
     * что ни один бэкенд не вернёт МОЛЧА МУСОР (нечисловое решение или решение,
     * не удовлетворяющее системе). Она НЕ делает бэкенды побитово неразличимыми и
     * не обещает, что оба бэкенда ОДИНАКОВО отбракуют плохо ОБУСЛОВЛЕННУЮ (но не
     * вырожденную) систему: у [ReferenceLinearAlgebra] есть СВОЯ, более ранняя
     * проверка ведущего элемента (её нельзя воспроизвести на LAPACK, который
     * ведущих элементов не показывает), поэтому на матрице с cond ~ 1e16 reference
     * может бросить исключение раньше, чем multik — при том что оба ответа честны.
     * На ТОЧНО вырожденном входе ЛЮБОГО масштаба оба бэкенда бросают
     * [IllegalStateException] — это и есть единая семантика вырожденности.
     */
    const val SINGULARITY_RELATIVE_TOLERANCE = 1e-10

    /**
     * Решение плотной СЛАУ A x = b через активный бэкенд.
     *
     * Входные A и b не изменяются. ЕДИНАЯ для всех бэкендов семантика
     * вырожденности обеспечивается ЗДЕСЬ, в фасаде, а не в бэкенде: бэкенды
     * имеют разные возможности диагностики (ручной LU видит ведущие элементы,
     * нативный LAPACK — нет), и единственный контракт, проверяемый одинаково для
     * любого из них, — числовой результат и малая невязка
     * (см. [SINGULARITY_RELATIVE_TOLERANCE]).
     *
     * @throws IllegalStateException при вырожденности: нечисловом решении либо
     *         невязке, несовместимой с машинной точностью.
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "solve: пустая матрица A" }
        // Квадратность и прямоугольность: бэкенды неявно исходят из n x n и одинаковой
        // длины строк; рваный вход дал бы либо AIOOBE в глубине, либо тихо неверное решение.
        require(a[0].size == a.size) {
            "solve: требуется квадратная A, получено ${a.size}x${a[0].size}"
        }
        for (i in a.indices) require(a[i].size == a.size) {
            "solve: рваная матрица A — строка $i длины ${a[i].size}, ожидалось ${a.size}"
        }
        require(a.size == b.size) { "solve: несогласованные размеры A(${a.size} строк) и b(${b.size})" }
        val x = Backends.active.solve(a, b)
        checkSolution(a, b, x)
        return x
    }

    /**
     * Постпроверка решения, ОБЩАЯ для всех бэкендов (см. [SINGULARITY_RELATIVE_TOLERANCE]).
     *
     * Невязка считается прямо здесь (без [matVec]), чтобы контроль не зависел от
     * того же бэкенда, чей ответ проверяется, и чтобы не создавать временные
     * нативные массивы на горячем пути (один проход по A, без аллокаций).
     */
    private fun checkSolution(a: Array<DoubleArray>, b: DoubleArray, x: DoubleArray) {
        val n = a.size
        for (i in x.indices) {
            if (x[i].isNaN() || x[i].isInfinite()) {
                error("solve: матрица вырождена — нечисловое решение x[$i]=${x[i]} (n=$n)")
            }
        }
        var matrixNorm = 0.0 // ||A||_inf
        var residual = 0.0 // ||Ax - b||_inf
        for (i in 0 until n) {
            val row = a[i]
            var rowSum = 0.0
            var ax = 0.0
            for (j in 0 until n) {
                rowSum += abs(row[j])
                ax += row[j] * x[j]
            }
            matrixNorm = maxOf(matrixNorm, rowSum)
            residual = maxOf(residual, abs(ax - b[i]))
        }
        val solutionNorm = normInf(x)
        val rhsNorm = normInf(b)
        val scale = maxOf(matrixNorm * solutionNorm, rhsNorm)
        if (residual > SINGULARITY_RELATIVE_TOLERANCE * scale) {
            error(
                "solve: матрица вырождена — невязка слишком велика: n=$n, " +
                    "||A||_inf=$matrixNorm, ||x||_inf=$solutionNorm, ||b||_inf=$rhsNorm, " +
                    "||Ax-b||_inf=$residual > $SINGULARITY_RELATIVE_TOLERANCE * $scale"
            )
        }
    }

    // --- Дешёвые скалярные/служебные операции (без бэкенда) ------------------
    // Реализации живут в [DenseOps] — едином источнике, общем с оракулом
    // [ReferenceLinearAlgebra]; здесь остаются только публичный контракт и валидация.

    /** Евклидова норма вектора. */
    fun norm2(x: DoubleArray): Double = DenseOps.norm2(x)

    /** Бесконечная (равномерная) норма вектора. */
    fun normInf(x: DoubleArray): Double = DenseOps.normInf(x)

    /**
     * Разложение Холецкого A = L L^T для симметричной положительно определённой A.
     * @return нижнетреугольная L или null, если A не положительно определена.
     */
    fun cholesky(a: Array<DoubleArray>): Array<DoubleArray>? = DenseOps.cholesky(a)

    /**
     * Симметрия: max|A - A^T|.
     *
     * Индексация `a[i][j]` идёт по `a.indices` по обоим индексам, т.е. квадратность требуется
     * по самому определению A^T; без проверки неквадратный/рваный вход давал бы
     * AIOOBE вместо диагностики. Проверки живут ИМЕННО ЗДЕСЬ, а не в [DenseOps]:
     * это контракт боевого фасада, а оракул [ReferenceLinearAlgebra] исторически
     * таких проверок не делает, и менять его поведение нельзя.
     */
    fun maxAsymmetry(a: Array<DoubleArray>): Double {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "maxAsymmetry: пустая матрица A" }
        require(a[0].size == a.size) {
            "maxAsymmetry: требуется квадратная A, получено ${a.size}x${a[0].size}"
        }
        for (i in a.indices) require(a[i].size == a.size) {
            "maxAsymmetry: рваная матрица A — строка $i длины ${a[i].size}, ожидалось ${a.size}"
        }
        return DenseOps.maxAsymmetry(a)
    }
}
