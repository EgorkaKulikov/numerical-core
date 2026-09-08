package numerics

import kotlin.math.abs
import numerics.backend.Backends
import numerics.backend.LinAlgBackend

/**
 * Линейная алгебра над плотными матрицами — единая точка входа библиотеки.
 * Основной тип матриц — [DenseMatrix] (плоский столбцовый формат); перегрузки на
 * [Array]<[DoubleArray]> сохранены как адаптеры с копией через [DenseMatrix.fromRows]
 * и [DenseMatrix.toRows].
 *
 * Тяжёлые операции (умножения и решение СЛАУ) выполняет реализация [LinAlgBackend],
 * переданная последним параметром; значение по умолчанию — [Backends.default]
 * (системная BLAS/LAPACK при доступности, иначе реализация на Java). Параметр, а не
 * глобальный переключатель: результат вызова не зависит от того, что успел выставить
 * сосед по JVM. Обращение к [Backends.default] не создаёт объектов и безопасно в горячем пути.
 * Дешёвые скалярные и служебные операции (нормы, проверка симметрии, разложение
 * Холецкого, конструкторы) реализованы в [DenseOps] без обращения к бэкенду.
 *
 * Проверки согласованности размеров и непустоты входа выполняются здесь, до вызова
 * бэкенда, и завершаются [IllegalArgumentException] с русским сообщением.
 */
object LinearAlgebra {

    // --- Тривиальные конструкторы (без бэкенда) ------------------------------

    /** Создаёт нулевую матрицу размера rows x cols. */
    fun zeros(rows: Int, cols: Int): Array<DoubleArray> = DenseOps.zeros(rows, cols)

    /** Единичная матрица размера n x n. */
    fun identity(n: Int): Array<DoubleArray> = DenseOps.identity(n)

    /** Создаёт нулевую матрицу rows x cols в плоском столбцовом формате. */
    fun zerosMatrix(rows: Int, cols: Int): DenseMatrix = DenseMatrix.zeros(rows, cols)

    /** Единичная матрица n x n в плоском столбцовом формате. */
    fun identityMatrix(n: Int): DenseMatrix = DenseMatrix.identity(n)

    // --- Тяжёлые операции над DenseMatrix: делегирование бэкенду --------------

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(
        a: DenseMatrix,
        x: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
        require(a.rows > 0 && a.cols > 0) { "matVec: пустая матрица A" }
        require(a.cols == x.size) { "matVec: несогласованные размеры A(${a.rows}x${a.cols}) и x(${x.size})" }
        return backend.matVec(a, x)
    }

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(
        a: DenseMatrix,
        y: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
        require(a.rows > 0 && a.cols > 0) { "matTransVec: пустая матрица A" }
        require(a.rows == y.size) { "matTransVec: несогласованные размеры A(${a.rows} строк) и y(${y.size})" }
        return backend.matTransVec(a, y)
    }

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(
        a: DenseMatrix,
        b: DenseMatrix,
        backend: LinAlgBackend = Backends.default(),
    ): DenseMatrix {
        require(a.rows > 0 && a.cols > 0) { "matMat: пустая матрица A" }
        require(b.rows > 0 && b.cols > 0) { "matMat: пустая матрица B" }
        require(a.cols == b.rows) { "matMat: несогласованные размеры A(${a.rows}x${a.cols}) и B(${b.rows}x${b.cols})" }
        return backend.matMat(a, b)
    }

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    fun atWa(
        a: DenseMatrix,
        w: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DenseMatrix {
        require(a.rows > 0 && a.cols > 0) { "atWa: пустая матрица A" }
        require(a.rows == w.size) { "atWa: несогласованные размеры A(${a.rows} строк) и w(${w.size})" }
        // B = diag(w)·A: строка i умножается на w[i]; затем A^T·B одним вызовом ядра.
        val scaled = a.copy()
        val d = scaled.data
        val m = a.rows
        for (j in 0 until a.cols) {
            val base = j * m
            for (i in 0 until m) d[base + i] *= w[i]
        }
        return backend.matTransMat(a, scaled)
    }

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(
        a: DenseMatrix,
        b: DenseMatrix,
        s: Double,
        @Suppress("UNUSED_PARAMETER") backend: LinAlgBackend = Backends.default(),
    ): DenseMatrix {
        require(a.rows == b.rows) { "addScaled: несогласованное число строк A(${a.rows}) и B(${b.rows})" }
        require(a.cols == b.cols) { "addScaled: несогласованное число столбцов A(${a.cols}) и B(${b.cols})" }
        val out = DoubleArray(a.data.size)
        val da = a.data
        val db = b.data
        for (k in out.indices) out[k] = da[k] + s * db[k]
        return DenseMatrix.fromColumnMajor(a.rows, a.cols, out)
    }

    /**
     * Относительный допуск на невязку решения СЛАУ, общий для всех реализаций.
     *
     * Критерий: `||Ax-b||_inf <= SINGULARITY_RELATIVE_TOLERANCE * max(||A||_inf*||x||_inf, ||b||_inf)`.
     *
     * Почему невязка, а не число обусловленности. Невязка — критерий, доступный для
     * любой реализации и дешёвый: O(n^2) против O(n^3) самого решения. Проверка по
     * обусловленности здесь была бы неверна по сути: регуляризованные задачи первого
     * рода штатно дают `cond ~ 1e10`, и отбраковывать их нельзя; невязка же там
     * ~1e-16 относительно масштаба — система решается обратно устойчиво.
     *
     * Откуда 1e-10. LU с частичным выбором обратно устойчив: `||Ax-b|| <= c*n*eps*||A||*||x||`
     * с eps=2.2e-16 и умеренным фактором роста; для n до нескольких сотен это ~1e-13.
     * Запас в три порядка оставлен сознательно: проверка обязана ловить мусор
     * (невязку порядка самого масштаба), а не различать оттенки качества двух честных LU.
     *
     * Зачем `max(..., ||b||_inf)`: при почти нулевом x произведение `||A||*||x||`
     * вырождается в ноль, и любая ошибка округления дала бы ложное «вырождение».
     *
     * Граница применимости: проверка гарантирует, что решатель не вернёт молча мусор
     * (нечисловое решение или решение, не удовлетворяющее системе); она не измеряет
     * прямую ошибку — для этого есть [solveDiagnosed].
     */
    const val SINGULARITY_RELATIVE_TOLERANCE = 1e-10

    /**
     * Решение плотной СЛАУ A x = b через переданный бэкенд.
     *
     * Входные A и b не изменяются. Единая семантика вырожденности обеспечивается
     * здесь: сначала отвергается нечисловой вход (NaN или бесконечность), затем
     * бэкенд сообщает о нулевом ведущем элементе, и наконец проверяется невязка
     * решения (см. [SINGULARITY_RELATIVE_TOLERANCE]).
     *
     * Постпроверка контролирует обратную ошибку, а не точность результата: на
     * численно вырожденной матрице невязка остаётся малой, тогда как прямая ошибка
     * растёт как `cond(A) · ε`. Эта величина здесь не измеряется сознательно (метод
     * лежит на горячем пути); когда она нужна — [solveDiagnosed], разбор различия
     * двух ошибок — [ForwardError].
     *
     * @throws IllegalStateException при нечисловом входе, вырожденности либо невязке,
     *         несовместимой с машинной точностью.
     */
    fun solve(
        a: DenseMatrix,
        b: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): DoubleArray {
        require(a.rows > 0 && a.cols > 0) { "solve: пустая матрица A" }
        require(a.isSquare) { "solve: требуется квадратная A, получено ${a.rows}x${a.cols}" }
        require(a.rows == b.size) { "solve: несогласованные размеры A(${a.rows} строк) и b(${b.size})" }
        for (v in a.data) if (!v.isFinite()) error("система содержит нечисловые значения (NaN или бесконечность)")
        for (v in b) if (!v.isFinite()) error("система содержит нечисловые значения (NaN или бесконечность)")
        val n = a.rows
        val x = backend.solve(a, DenseMatrix.fromColumnMajor(n, 1, b.copyOf())).data
        checkSolution(a, b, x, backend)
        return x
    }

    /**
     * Источник оценки `cond`, по которой [solveDiagnosed] строит границу прямой ошибки.
     *
     * Перечисление, а не булев флаг: выбор здесь не оптимизация, а разные
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
         * требует симметричной `A` и дороже, но различает «конечного `cond`
         * нет вовсе» (`σ_min = 0`, режим [ForwardError.NoFiniteBound]) и «`cond` велико,
         * но конечно» — различение, недоступное пути [INVERSION].
         */
        SYMMETRIC_SPECTRUM,
    }

    /**
     * Решение СЛАУ вместе с оценкой его прямой ошибки.
     *
     * @property x тот же вектор, что вернул бы [LinearAlgebra.solve] на тех же входных данных
     *   — диагностика не меняет вычислений и не уточняет решение.
     * @property forwardError граница относительной прямой ошибки либо явное свидетельство
     *   того, что границы не существует; см. [ForwardError].
     */
    class DiagnosedSolution(val x: DoubleArray, val forwardError: ForwardError)

    /**
     * Решает `A x = b` и оценивает прямую ошибку полученного решения.
     *
     * Постпроверка внутри [solve] контролирует обратную ошибку — невязку относительно
     * масштаба системы; у LU с частичным выбором она мала всегда, включая численно
     * вырожденные матрицы, и ничего не говорит о точности. Прямая ошибка растёт как
     * `cond(A) · ε` и на плохо обусловленной системе съедает значащие цифры.
     *
     * Диагностика опциональна и не встроена в [solve]: она стоит отдельных O(n³),
     * а [solve] вызывается на горячем пути (в частности, `n` раз внутри
     * [Conditioning.inverse]; встроенная диагностика дала бы бесконечную рекурсию).
     *
     * Функция не бросает исключение из-за большого `cond` и не подменяет решение:
     * большое `cond` — штатный режим регуляризованных задач. Решение, доверять ли
     * числу, принимает вызывающий. На точно вырожденной системе бросается
     * [IllegalStateException] тем же контрактом, что и [solve].
     *
     * @param source чем оценивать `cond`; см. [ConditionSource].
     * @param tolerance порог достоверности по невязке обращения; читается только
     *        при [ConditionSource.INVERSION].
     * @throws IllegalStateException при вырожденности — тем же контрактом, что и [solve].
     * @throws IllegalArgumentException при [ConditionSource.SYMMETRIC_SPECTRUM] и несимметричной `A`.
     */
    fun solveDiagnosed(
        a: DenseMatrix,
        b: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
        source: ConditionSource = ConditionSource.INVERSION,
        tolerance: Double = Conditioning.INVERSION_RESIDUAL_TOLERANCE,
    ): DiagnosedSolution {
        val x = solve(a, b, backend)
        val rows = a.toRows()
        val omega = Conditioning.relativeBackwardError(rows, b, x)
        val forward = when (source) {
            ConditionSource.INVERSION -> Conditioning.forwardError(Conditioning.conditionInf(rows, tolerance), omega)
            ConditionSource.SYMMETRIC_SPECTRUM -> Conditioning.forwardErrorSymmetric(rows, omega)
        }
        return DiagnosedSolution(x, forward)
    }

    /** Постпроверка решения, общая для всех реализаций (см. [SINGULARITY_RELATIVE_TOLERANCE]). */
    private fun checkSolution(a: DenseMatrix, b: DoubleArray, x: DoubleArray, backend: LinAlgBackend) {
        val n = a.rows
        for (i in x.indices) {
            if (x[i].isNaN() || x[i].isInfinite()) {
                error("solve: матрица вырождена — нечисловое решение x[$i]=${x[i]} (n=$n)")
            }
        }
        val ax = backend.matVec(a, x)
        var residual = 0.0
        for (i in 0 until n) residual = maxOf(residual, abs(ax[i] - b[i]))
        var matrixNorm = 0.0
        val d = a.data
        for (i in 0 until n) {
            var rowSum = 0.0
            for (j in 0 until n) rowSum += abs(d[i + j * n])
            matrixNorm = maxOf(matrixNorm, rowSum)
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

    /** Евклидова норма вектора. */
    fun norm2(x: DoubleArray): Double = DenseOps.norm2(x)

    /** Бесконечная (равномерная) норма вектора. */
    fun normInf(x: DoubleArray): Double = DenseOps.normInf(x)

    /**
     * Разложение Холецкого A = L L^T для симметричной положительно определённой A.
     * @return нижнетреугольная L или null, если A не положительно определена.
     */
    fun cholesky(a: DenseMatrix): DenseMatrix? = DenseOps.cholesky(a.toRows())?.let { DenseMatrix.fromRows(it) }

    /** Симметрия: max|A - A^T|; требует непустую квадратную A. */
    fun maxAsymmetry(a: DenseMatrix): Double {
        require(a.rows > 0 && a.cols > 0) { "maxAsymmetry: пустая матрица A" }
        require(a.isSquare) { "maxAsymmetry: требуется квадратная A, получено ${a.rows}x${a.cols}" }
        return DenseOps.maxAsymmetry(a.toRows())
    }

    // --- Адаптеры над Array<DoubleArray> ---------------------------------------
    // Рваный массив отвергает DenseMatrix.fromRows (IllegalArgumentException с номером строки).

    private fun rows(a: Array<DoubleArray>, op: String): DenseMatrix {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "$op: пустая матрица A" }
        return DenseMatrix.fromRows(a)
    }

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray, backend: LinAlgBackend = Backends.default()): DoubleArray =
        matVec(rows(a, "matVec"), x, backend)

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray, backend: LinAlgBackend = Backends.default()): DoubleArray =
        matTransVec(rows(a, "matTransVec"), y, backend)

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    @Deprecated(
        "Используйте перегрузку с DenseMatrix",
        ReplaceWith("matMat(DenseMatrix.fromRows(a), DenseMatrix.fromRows(b), backend)"),
        DeprecationLevel.WARNING,
    )
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>, backend: LinAlgBackend = Backends.default()): Array<DoubleArray> {
        require(b.isNotEmpty() && b[0].isNotEmpty()) { "matMat: пустая матрица B" }
        return matMat(rows(a, "matMat"), DenseMatrix.fromRows(b), backend).toRows()
    }

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    @Deprecated(
        "Используйте перегрузку с DenseMatrix",
        ReplaceWith("atWa(DenseMatrix.fromRows(a), w, backend)"),
        DeprecationLevel.WARNING,
    )
    fun atWa(a: Array<DoubleArray>, w: DoubleArray, backend: LinAlgBackend = Backends.default()): Array<DoubleArray> =
        atWa(rows(a, "atWa"), w, backend).toRows()

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    @Deprecated(
        "Используйте перегрузку с DenseMatrix",
        ReplaceWith("addScaled(DenseMatrix.fromRows(a), DenseMatrix.fromRows(b), s, backend)"),
        DeprecationLevel.WARNING,
    )
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double, backend: LinAlgBackend = Backends.default()): Array<DoubleArray> {
        require(a.size == b.size) { "addScaled: несогласованное число строк A(${a.size}) и B(${b.size})" }
        return addScaled(DenseMatrix.fromRows(a), DenseMatrix.fromRows(b), s, backend).toRows()
    }

    /**
     * Решение плотной СЛАУ A x = b; контракт тот же, что у перегрузки над [DenseMatrix],
     * включая проверку нечислового входа и постпроверку невязки.
     * @throws IllegalStateException при вырожденности.
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray, backend: LinAlgBackend = Backends.default()): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "solve: пустая матрица A" }
        require(a[0].size == a.size) { "solve: требуется квадратная A, получено ${a.size}x${a[0].size}" }
        for (i in a.indices) require(a[i].size == a.size) {
            "solve: рваная матрица A — строка $i длины ${a[i].size}, ожидалось ${a.size}"
        }
        return solve(DenseMatrix.fromRows(a), b, backend)
    }

    /** Решает A x = b и оценивает прямую ошибку; контракт тот же, что у перегрузки над [DenseMatrix]. */
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
     * Разложение Холецкого A = L L^T для симметричной положительно определённой A.
     * @return нижнетреугольная L или null, если A не положительно определена.
     */
    @Deprecated(
        "Используйте перегрузку с DenseMatrix",
        ReplaceWith("cholesky(DenseMatrix.fromRows(a))"),
        DeprecationLevel.WARNING,
    )
    fun cholesky(a: Array<DoubleArray>): Array<DoubleArray>? = DenseOps.cholesky(a)

    /** Симметрия: max|A - A^T|; требует непустую квадратную нерваную A. */
    fun maxAsymmetry(a: Array<DoubleArray>): Double {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "maxAsymmetry: пустая матрица A" }
        require(a[0].size == a.size) { "maxAsymmetry: требуется квадратная A, получено ${a.size}x${a[0].size}" }
        for (i in a.indices) require(a[i].size == a.size) {
            "maxAsymmetry: рваная матрица A — строка $i длины ${a[i].size}, ожидалось ${a.size}"
        }
        return DenseOps.maxAsymmetry(a)
    }
}
