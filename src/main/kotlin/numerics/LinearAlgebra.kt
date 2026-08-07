package numerics

import kotlin.math.abs
import numerics.backend.Backends
import numerics.backend.LinAlgBackend

/**
 * Линейная алгебра над [Array]<[DoubleArray]> — тонкий фасад над подключаемым
 * бэкендом (SPI).
 *
 * Публичный API стабилен (массивы Kotlin) и не зависит от выбранной реализации.
 * Тяжёлые операции (умножения и решение СЛАУ) делегируются бэкенду, переданному
 * ЯВНО последним параметром; его значение по умолчанию —
 * [numerics.backend.Backends.default] (нативный multik/OpenBLAS-бэкенд, с
 * автоматическим откатом на чистый JVM при отсутствии нативной библиотеки).
 * Параметр, а не глобальный переключатель: вызов больше не зависит от того, что
 * успел выставить сосед по JVM. Обращение к [numerics.backend.Backends.default]
 * не создаёт объектов, поэтому дефолт безопасен в горячем пути.
 * Дешёвые скалярные/служебные операции (нормы, проверка
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
    fun matVec(
        a: Array<DoubleArray>,
        x: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matVec: пустая матрица A" }
        require(a[0].size == x.size) { "matVec: несогласованные размеры A(${a.size}x${a[0].size}) и x(${x.size})" }
        return backend.matVec(a, x)
    }

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(
        a: Array<DoubleArray>,
        y: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matTransVec: пустая матрица A" }
        require(a.size == y.size) { "matTransVec: несогласованные размеры A(${a.size} строк) и y(${y.size})" }
        return backend.matTransVec(a, y)
    }

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(
        a: Array<DoubleArray>,
        b: Array<DoubleArray>,
        backend: LinAlgBackend = Backends.default(),
    ): Array<DoubleArray> {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matMat: пустая матрица A" }
        require(b.isNotEmpty() && b[0].isNotEmpty()) { "matMat: пустая матрица B" }
        require(a[0].size == b.size) { "matMat: несогласованные размеры A(${a.size}x${a[0].size}) и B(${b.size}x${b[0].size})" }
        return backend.matMat(a, b)
    }

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    fun atWa(
        a: Array<DoubleArray>,
        w: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): Array<DoubleArray> {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "atWa: пустая матрица A" }
        require(a.size == w.size) { "atWa: несогласованные размеры A(${a.size} строк) и w(${w.size})" }
        return backend.atWa(a, w)
    }

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(
        a: Array<DoubleArray>,
        b: Array<DoubleArray>,
        s: Double,
        backend: LinAlgBackend = Backends.default(),
    ): Array<DoubleArray> {
        require(a.size == b.size) { "addScaled: несогласованное число строк A(${a.size}) и B(${b.size})" }
        require(a.isEmpty() || a[0].size == b[0].size) {
            "addScaled: несогласованное число столбцов A(${a[0].size}) и B(${b[0].size})"
        }
        return backend.addScaled(a, b, s)
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
     * Решение плотной СЛАУ A x = b через переданный бэкенд.
     *
     * Входные A и b не изменяются. ЕДИНАЯ для всех бэкендов семантика
     * вырожденности обеспечивается ЗДЕСЬ, в фасаде, а не в бэкенде: бэкенды
     * имеют разные возможности диагностики (ручной LU видит ведущие элементы,
     * нативный LAPACK — нет), и единственный контракт, проверяемый одинаково для
     * любого из них, — числовой результат и малая невязка
     * (см. [SINGULARITY_RELATIVE_TOLERANCE]).
     *
     * ЧТО ЭТОТ МЕТОД НЕ ОБЕЩАЕТ. Постпроверка контролирует ОБРАТНУЮ ошибку,
     * а не точность результата: на численно вырожденной матрице невязка остаётся
     * малой (измерено 3.97e-16), тогда как ПРЯМАЯ ошибка растёт как `cond(A) · ε`
     * и может съесть все значащие цифры. Здесь эта величина не измеряется СОЗНАТЕЛЬНО
     * (метод лежит на горячем пути сборки матриц); когда она нужна —
     * [solveDiagnosed], разбор различия двух ошибок — [ForwardError].
     *
     * @throws IllegalStateException при вырожденности: нечисловом решении либо
     *         невязке, несовместимой с машинной точностью.
     */
    fun solve(
        a: Array<DoubleArray>,
        b: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
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
        val x = backend.solve(a, b)
        checkSolution(a, b, x)
        return x
    }

    /**
     * Источник оценки `cond`, по которой [solveDiagnosed] строит границу прямой ошибки.
     *
     * Перечисление, а не булев флаг: выбор здесь НЕ оптимизация, а разные
     * математические гарантии — и разные требования к входу.
     */
    enum class ConditionSource {
        /**
         * Через явное обращение ([Conditioning.conditionInf]): O(n³), без требований к `A`.
         * На почти вырожденной матрице сама оценка теряет достоверность, и результатом
         * становится [ForwardError.Unreliable] — честное «числа нет».
         */
        INVERSION,

        /**
         * Через спектр методом вращений Якоби ([Conditioning.conditionSymmetric]):
         * требует СИММЕТРИЧНОЙ `A` и дороже, но различает «конечного `cond`
         * нет вовсе» (`σ_min = 0`, режим [ForwardError.NoFiniteBound]) и «`cond` велико,
         * но конечно» — различение, недоступное пути [INVERSION].
         */
        SYMMETRIC_SPECTRUM,
    }

    /**
     * Решение СЛАУ вместе с оценкой его ПРЯМОЙ ошибки.
     *
     * @property x тот же вектор, что вернул бы [LinearAlgebra.solve] на тех же входных данных
     *   — диагностика НЕ меняет вычислений и не уточняет решение.
     * @property forwardError граница относительной прямой ошибки либо явное свидетельство
     *   того, что границы не существует; см. [ForwardError].
     */
    class DiagnosedSolution(val x: DoubleArray, val forwardError: ForwardError)

    /**
     * Решает `A x = b` И ОЦЕНИВАЕТ ПРЯМУЮ ОШИБКУ полученного решения.
     *
     * ПОЧЕМУ ЭТОГО НЕ ДЕЛАЕТ [solve]. Постпроверка внутри [solve] контролирует
     * ОБРАТНУЮ ошибку — невязку `‖Ax − b‖∞` относительно масштаба системы
     * (см. [SINGULARITY_RELATIVE_TOLERANCE]). У LU с частичным выбором эта величина
     * мала ВСЕГДА — включая численно вырожденные матрицы (измерено 3.97e-16),
     * поэтому постпроверка ловит МУСОР, но НИЧЕГО НЕ ГОВОРИТ О ТОЧНОСТИ.
     * Прямая ошибка растёт как `cond(A) · ε` и на плохо обусловленной системе съедает
     * все значащие цифры: на выходе [solve] режим «результат точен» и режим
     * «результат бессмыслен» для вызывающего неразличимы. Различие двух ошибок
     * разобрано в KDoc [ForwardError].
     *
     * ДИАГНОСТИКА ОПЦИОНАЛЬНА И НЕ ВСТРОЕНА В [solve]. Она стоит отдельных
     * O(n³) — столько же, сколько само решение, то есть удваивает цену, а [solve]
     * вызывается на горячем пути сборки матриц (в частности, `n` раз внутри
     * самого [Conditioning.inverse]). Встроенная диагностика дала бы ещё и
     * бесконечную рекурсию. Все существующие вызовы [solve] остаются без изменений.
     *
     * НЕ ОТБРАКОВКА. Функция НЕ бросает исключение из-за большого `cond` и не
     * подменяет решение: большое `cond` здесь — штатный режим метода, а не
     * ошибка (задача F1 с `alpha = 1e-10` штатно даёт `cond ≈ 2.6e10`; почему
     * отбраковка по обусловленности была бы НЕВЕРНА по сути — в KDoc
     * [SINGULARITY_RELATIVE_TOLERANCE]). Решение, доверять ли числу, принимает
     * вызывающий.
     *
     * КОНТРАКТ НА ВЫРОЖДЕННОСТИ СОХРАНЁН: решение идёт через [solve], поэтому
     * на точно вырожденной системе бросается [IllegalStateException] до того, как
     * начнётся диагностика.
     *
     * @param source чем оценивать `cond`; см. [ConditionSource].
     * @param tolerance порог достоверности по невязке обращения; читается только
     *        при [ConditionSource.INVERSION].
     * @throws IllegalStateException при вырожденности — тем же контрактом, что и [solve].
     * @throws IllegalArgumentException при [ConditionSource.SYMMETRIC_SPECTRUM] и несимметричной `A`.
     */
    fun solveDiagnosed(
        a: Array<DoubleArray>,
        b: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
        source: ConditionSource = ConditionSource.INVERSION,
        tolerance: Double = Conditioning.INVERSION_RESIDUAL_TOLERANCE,
    ): DiagnosedSolution {
        val x = solve(a, b, backend)
        val omega = Conditioning.relativeBackwardError(a, b, x)
        val forward = when (source) {
            ConditionSource.INVERSION -> Conditioning.forwardError(Conditioning.conditionInf(a, tolerance), omega)
            ConditionSource.SYMMETRIC_SPECTRUM -> Conditioning.forwardErrorSymmetric(a, omega)
        }
        return DiagnosedSolution(x, forward)
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
