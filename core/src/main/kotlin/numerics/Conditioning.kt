package numerics

import kotlin.math.abs
import kotlin.math.log10
import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.backend.MatrixNorm

/**
 * Оценка числа обусловленности плотной матрицы вместе с признаком её достоверности.
 *
 * Число обусловленности само вычисляется численно, и на почти вырожденных матрицах
 * его значение недостоверно: прямая ошибка обращения растёт как `cond(A)·ε`, и при
 * `cond` порядка `1/ε` в столбцах `A⁻¹` не остаётся верных разрядов. Невязка
 * `‖A·A⁻¹ − I‖∞` измеряет именно эту потерю, поэтому оценка сопровождается ею,
 * а достоверность проверяется через [isReliable].
 *
 * @property condInf оценка числа обусловленности; [Double.POSITIVE_INFINITY], если
 *   матрица признана вырожденной.
 * @property inversionResidual невязка обращения `‖A·A⁻¹ − I‖∞`; [Double.POSITIVE_INFINITY],
 *   если обращение не удалось; `0.0`, если оценка получена без обращения
 *   ([Conditioning.conditionEstimate]).
 * @property tolerance порог невязки, при котором оценка считается достоверной; строго положителен.
 */
public data class ConditionEstimate(
    val condInf: Double,
    val inversionResidual: Double,
    val tolerance: Double,
) {
    init {
        require(tolerance > 0.0) { "ConditionEstimate: требуется tolerance > 0, получено $tolerance" }
    }

    /** Достоверна ли оценка: значение конечно и невязка обращения не превышает [tolerance]. */
    val isReliable: Boolean
        get() = condInf.isFinite() && inversionResidual <= tolerance

    /** [condInf], если оценка достоверна, иначе `null`. */
    public fun valueOrNull(): Double? = if (isReliable) condInf else null
}

/**
 * Способ получения числа обусловленности, по которому [LinearAlgebra.solveDiagnosed]
 * строит границу прямой ошибки. Варианты различаются гарантиями, стоимостью и
 * требованиями к матрице.
 */
public enum class ConditionSource {
    /**
     * Явное обращение матрицы ([Conditioning.conditionInf]): O(n³), без требований к `A`,
     * достоверность контролируется невязкой обращения.
     */
    INVERSION,

    /**
     * Спектр симметричной матрицы ([Conditioning.conditionSymmetric]): требует симметричной `A`;
     * различает вырожденную матрицу (`min|λ| = 0`) и матрицу с большим, но конечным `cond`.
     */
    SYMMETRIC_SPECTRUM,

    /**
     * Оценка LAPACK по LU-разложению ([Conditioning.conditionEstimate]): O(n²) сверх
     * разложения, без требований к `A`; даёт оценку снизу с точностью до множителя порядка единиц.
     */
    ESTIMATE,
}

/**
 * Оценка относительной прямой ошибки решения СЛАУ вместе с суждением о том,
 * существует ли она и можно ли ей верить.
 *
 * Обратная ошибка `ω = ‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)` показывает, какую систему
 * метод решил точно; у LU с частичным выбором она мала и на почти вырожденной матрице.
 * Прямая ошибка `‖x − x*‖∞ / ‖x*‖∞` ограничена произведением `cond(A)·ω` и на
 * плохо обусловленной системе может съесть все значащие цифры. Малая обратная
 * ошибка сама по себе не означает точного результата.
 *
 * Три режима разделены конструкторами: получить число можно только явным
 * сопоставлением либо через [relativeBoundOrNull], который возвращает `null`, где числа нет.
 */
public sealed interface ForwardError {
    /** Измеренная относительная обратная ошибка; доступна во всех режимах. */
    public val backwardError: Double

    /**
     * Оценка существует и достоверна: `‖x − x*‖∞ / ‖x*‖∞ <= [relativeBound]`.
     *
     * @property cond использованное число обусловленности.
     * @property relativeBound произведение `cond · backwardError`.
     */
    public data class Bounded(
        override val backwardError: Double,
        val cond: Double,
        val relativeBound: Double,
    ) : ForwardError

    /**
     * Конечной границы нет: матрица численно вырождена — обращение не удалось либо
     * спектральная оценка дала нулевое собственное значение.
     */
    public data class NoFiniteBound(override val backwardError: Double) : ForwardError

    /**
     * Оценка `cond` конечна, но недостоверна ([ConditionEstimate.isReliable] равно `false`).
     *
     * @property condition сама недостоверная оценка — невязка обращения лежит в ней.
     */
    public data class Unreliable(
        override val backwardError: Double,
        val condition: ConditionEstimate,
    ) : ForwardError

    /** Граница прямой ошибки или `null` в режимах [NoFiniteBound] и [Unreliable]. */
    public fun relativeBoundOrNull(): Double? = (this as? Bounded)?.relativeBound

    /**
     * Сколько десятичных разрядов результата заведомо уцелело: `−log10(границы)`,
     * обрезанное снизу нулём и сверху 16 (полная мантисса double).
     * `null` там же, где `null` у [relativeBoundOrNull].
     */
    public fun survivingDigitsOrNull(): Double? {
        val bound = relativeBoundOrNull() ?: return null
        if (bound <= 0.0) return 16.0
        return minOf(16.0, maxOf(0.0, -log10(bound)))
    }
}

/**
 * Числа обусловленности плотных матриц и оценка прямой ошибки решения СЛАУ.
 *
 * Все вычисления выполняет [LinAlgBackend]: обращение — решение A·X = I по LU-разложению (dgetrf + dgetrs),
 * оценка обусловленности — dgecon, спектр симметричной матрицы — dsyev. Основные
 * перегрузки работают с [DenseMatrix]; перегрузки на [Array]<[DoubleArray]> — адаптеры
 * с копией через [DenseMatrix.fromRows].
 *
 * Оценка через явное обращение ([conditionInf]) достоверна на хорошо обусловленных
 * матрицах и теряет смысл на почти вырожденных: `A⁻¹` вычисляется тем же LU, чья
 * прямая ошибка растёт как `cond(A)·ε`. Невязка `‖A·A⁻¹ − I‖∞` показывает эту потерю
 * напрямую, поэтому результат всегда содержит её ([ConditionEstimate]). Для симметричных
 * матриц надёжнее спектральная оценка [conditionSymmetric]: ортогональные преобразования
 * не обращают матрицу и на вырожденной честно дают нулевое собственное значение.
 */
public object Conditioning {

    /**
     * Порог достоверности оценки [conditionInf] по невязке обращения `‖A·A⁻¹ − I‖∞`.
     * Соответствует потере половины значащих разрядов: при большей невязке в оценке
     * `cond` не остаётся и половины верных цифр.
     */
    public const val INVERSION_RESIDUAL_TOLERANCE: Double = 1e-8

    /** Строчная норма матрицы `‖A‖∞ = max_i Σ_j |a_ij|`; матрица должна быть непустой. */
    public fun matrixNormInf(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): Double {
        Shapes.requireNonEmpty(a, "матрица A")
        return backend.norm(a, MatrixNorm.INF)
    }

    /** Строчная норма матрицы над массивом строк; см. перегрузку с [DenseMatrix]. */
    public fun matrixNormInf(a: Array<DoubleArray>): Double = matrixNormInf(rows(a))

    /**
     * Обращение квадратной матрицы через LU-разложение.
     *
     * @return `A⁻¹` или `null`, если матрица признана вырожденной либо результат содержит
     *   нечисловые значения. Достоверность конечного результата не гарантируется — её
     *   измеряет [inversionResidual].
     */
    public fun inverse(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): DenseMatrix? {
        Shapes.requireSquare(a, "матрица A")
        val inv = backend.inverse(a) ?: return null
        for (v in inv.data) if (!v.isFinite()) return null
        return inv
    }

    /** Обращение над массивом строк; см. перегрузку с [DenseMatrix]. */
    @Deprecated(
        "Используйте перегрузку с DenseMatrix",
        ReplaceWith("inverse(DenseMatrix.fromRows(a))"),
        DeprecationLevel.WARNING,
    )
    public fun inverse(a: Array<DoubleArray>): Array<DoubleArray>? = inverse(rows(a))?.toRows()

    /** Невязка обращения `‖A·B − I‖∞` — мера того, сколько разрядов уцелело в `B ≈ A⁻¹`. */
    public fun inversionResidual(a: DenseMatrix, inv: DenseMatrix, backend: LinAlgBackend = Backends.default()): Double {
        Shapes.requireSquare(a, "матрица A")
        Shapes.requireSquare(inv, "матрица A⁻¹")
        Shapes.requireSameShape(a, inv)
        val r = backend.matMat(a, inv)
        val n = a.rows
        val d = r.data
        for (i in 0 until n) d[i + i * n] -= 1.0
        return backend.norm(r, MatrixNorm.INF)
    }

    /** Невязка обращения над массивами строк; см. перегрузку с [DenseMatrix]. */
    public fun inversionResidual(a: Array<DoubleArray>, inv: Array<DoubleArray>): Double =
        inversionResidual(rows(a), rows(inv))

    /**
     * Оценка `cond∞(A) = ‖A‖∞ · ‖A⁻¹‖∞` через явное обращение, вместе с невязкой обращения.
     *
     * На вырожденной матрице исключение не бросается: возвращается бесконечная оценка
     * с бесконечной невязкой, и [ConditionEstimate.isReliable] ложно.
     *
     * @param tolerance порог достоверности по невязке обращения; строго положителен.
     * @throws IllegalArgumentException если матрица пуста, не квадратна или `tolerance <= 0`.
     */
    public fun conditionInf(
        a: DenseMatrix,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
        backend: LinAlgBackend = Backends.default(),
    ): ConditionEstimate {
        require(tolerance > 0.0) { "conditionInf: требуется tolerance > 0, получено $tolerance" }
        Shapes.requireSquare(a, "матрица A")
        val inv = inverse(a, backend)
            ?: return ConditionEstimate(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, tolerance)
        val cond = matrixNormInf(a, backend) * matrixNormInf(inv, backend)
        return ConditionEstimate(cond, inversionResidual(a, inv, backend), tolerance)
    }

    /** Оценка обусловленности через обращение над массивом строк; см. перегрузку с [DenseMatrix]. */
    public fun conditionInf(
        a: Array<DoubleArray>,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate = conditionInf(rows(a), tolerance)

    /**
     * Оценка числа обусловленности в норме-1 средствами LAPACK (dgecon) по LU-разложению.
     *
     * Даёт оценку снизу, обычно в пределах множителя порядка единиц от точного значения;
     * стоимость O(n²) после факторизации против O(n³) для явного обращения.
     * [ConditionEstimate.inversionResidual] равна нулю, потому что обращение не выполняется,
     * и для конечной оценки [ConditionEstimate.isReliable] истинно — достоверность здесь
     * обеспечивается алгоритмом. На вырожденной матрице возвращается бесконечная оценка
     * (её [ConditionEstimate.valueOrNull] даёт `null`).
     *
     * @throws IllegalArgumentException если матрица пуста, не квадратна или `tolerance <= 0`.
     */
    public fun conditionEstimate(
        a: DenseMatrix,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
        backend: LinAlgBackend = Backends.default(),
    ): ConditionEstimate {
        require(tolerance > 0.0) { "conditionEstimate: требуется tolerance > 0, получено $tolerance" }
        Shapes.requireSquare(a, "матрица A")
        val lu = backend.luFactor(a)
        if (lu.isSingular) return ConditionEstimate(Double.POSITIVE_INFINITY, 0.0, tolerance)
        val rcond = backend.reciprocalCondition1(lu, backend.norm(a, MatrixNorm.ONE))
        val cond = if (rcond > 0.0) 1.0 / rcond else Double.POSITIVE_INFINITY
        return ConditionEstimate(cond, 0.0, tolerance)
    }

    /**
     * Собственные значения симметричной матрицы по возрастанию (dsyev).
     *
     * @throws IllegalArgumentException если матрица пуста, не квадратна или её асимметрия
     *   превышает `1e-12 · ‖A‖∞`.
     */
    public fun symmetricEigenvalues(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): DoubleArray {
        Shapes.requireSquare(a, "матрица A")
        Shapes.requireSymmetric(a, backend.norm(a, MatrixNorm.INF), "матрица A")
        return backend.symmetricEigenvalues(a)
    }

    /** Собственные значения над массивом строк; см. перегрузку с [DenseMatrix]. */
    public fun symmetricEigenvalues(a: Array<DoubleArray>): DoubleArray = symmetricEigenvalues(rows(a))

    /**
     * Наименьшее по модулю собственное значение симметричной матрицы — её `σ_min`.
     * Ровно `0.0` означает численную вырожденность.
     */
    public fun smallestMagnitudeEigenvalue(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): Double {
        var m = Double.POSITIVE_INFINITY
        for (v in symmetricEigenvalues(a, backend)) m = minOf(m, abs(v))
        return m
    }

    /** Наименьшее по модулю собственное значение над массивом строк; см. перегрузку с [DenseMatrix]. */
    public fun smallestMagnitudeEigenvalue(a: Array<DoubleArray>): Double = smallestMagnitudeEigenvalue(rows(a))

    /**
     * Спектральное число обусловленности симметричной матрицы `max|λ_i| / min|λ_i|`.
     * Возвращает [Double.POSITIVE_INFINITY] при `min|λ_i| = 0`.
     */
    public fun conditionSymmetric(a: DenseMatrix, backend: LinAlgBackend = Backends.default()): Double {
        var lo = Double.POSITIVE_INFINITY
        var hi = 0.0
        for (v in symmetricEigenvalues(a, backend)) {
            lo = minOf(lo, abs(v))
            hi = maxOf(hi, abs(v))
        }
        return if (lo == 0.0) Double.POSITIVE_INFINITY else hi / lo
    }

    /** Спектральное число обусловленности над массивом строк; см. перегрузку с [DenseMatrix]. */
    public fun conditionSymmetric(a: Array<DoubleArray>): Double = conditionSymmetric(rows(a))

    /**
     * Относительная обратная ошибка решения `A x = b`:
     * `ω = ‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)`.
     *
     * Нормировка совпадает с постпроверкой [LinearAlgebra.solve] по порогу
     * [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE]. Величина измеряется, а не оценивается,
     * и потому достоверна всегда; прямую ошибку она не ограничивает — множителем служит
     * `cond(A)`, см. [ForwardError]. При нулевом масштабе (`A = 0`, `b = 0`) возвращается `0.0`.
     *
     * @throws IllegalArgumentException если матрица пуста, не квадратна или длины `b`, `x`
     *   не равны её порядку.
     */
    public fun relativeBackwardError(
        a: DenseMatrix,
        b: DoubleArray,
        x: DoubleArray,
        backend: LinAlgBackend = Backends.default(),
    ): Double {
        Shapes.requireSquare(a, "матрица A")
        val n = a.rows
        Shapes.requireVectorLength(b, n, "b")
        Shapes.requireVectorLength(x, n, "x")
        val r = backend.matVec(a, x)
        for (i in 0 until n) r[i] -= b[i]
        val scale = maxOf(backend.norm(a, MatrixNorm.INF) * LinearAlgebra.normInf(x), LinearAlgebra.normInf(b))
        return if (scale == 0.0) 0.0 else LinearAlgebra.normInf(r) / scale
    }

    /** Обратная ошибка над массивом строк; см. перегрузку с [DenseMatrix]. */
    public fun relativeBackwardError(a: Array<DoubleArray>, b: DoubleArray, x: DoubleArray): Double =
        relativeBackwardError(rows(a), b, x)

    /**
     * Граница относительной прямой ошибки `cond·ω` по готовой оценке обусловленности
     * и измеренной обратной ошибке `ω`.
     *
     * Функция ничего не вычисляет заново: она переводит пару «оценка `cond` + обратная
     * ошибка» в один из трёх режимов [ForwardError], не позволяя недостоверному `cond`
     * превратиться в число.
     *
     * @param backwardError результат [relativeBackwardError]; требуется конечным и неотрицательным.
     */
    public fun forwardError(condition: ConditionEstimate, backwardError: Double): ForwardError {
        require(backwardError.isFinite() && backwardError >= 0.0) {
            "forwardError: обратная ошибка должна быть конечной и неотрицательной, получено $backwardError"
        }
        return when {
            !condition.condInf.isFinite() -> ForwardError.NoFiniteBound(backwardError)
            !condition.isReliable -> ForwardError.Unreliable(backwardError, condition)
            else -> ForwardError.Bounded(backwardError, condition.condInf, condition.condInf * backwardError)
        }
    }

    /**
     * Граница относительной прямой ошибки для симметричной матрицы через спектральную
     * оценку [conditionSymmetric].
     *
     * Различает два факта: конечного числа обусловленности нет вовсе (`min|λ| = 0`,
     * режим [ForwardError.NoFiniteBound]) и `cond` велико, но конечно
     * ([ForwardError.Bounded] с большой границей). Режим [ForwardError.Unreliable]
     * не возвращается никогда.
     *
     * @throws IllegalArgumentException если матрица не симметрична.
     */
    public fun forwardErrorSymmetric(
        a: DenseMatrix,
        backwardError: Double,
        backend: LinAlgBackend = Backends.default(),
    ): ForwardError {
        require(backwardError.isFinite() && backwardError >= 0.0) {
            "forwardErrorSymmetric: обратная ошибка должна быть конечной и неотрицательной, получено $backwardError"
        }
        val cond = conditionSymmetric(a, backend)
        return if (!cond.isFinite()) {
            ForwardError.NoFiniteBound(backwardError)
        } else {
            ForwardError.Bounded(backwardError, cond, cond * backwardError)
        }
    }

    /** Граница прямой ошибки для симметричной матрицы над массивом строк; см. перегрузку с [DenseMatrix]. */
    public fun forwardErrorSymmetric(a: Array<DoubleArray>, backwardError: Double): ForwardError =
        forwardErrorSymmetric(rows(a), backwardError)

    /** Адаптер массива строк: пустой массив отвергается здесь, рваный — в [DenseMatrix.fromRows]. */
    private fun rows(a: Array<DoubleArray>): DenseMatrix {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "матрица A не должна быть пустой" }
        return DenseMatrix.fromRows(a)
    }

    // Перегрузки с NumericsContext используют его реализацию линейной алгебры.

    /** То же, что [inverse], с реализацией линейной алгебры из [context]. */
    public fun inverse(a: DenseMatrix, context: NumericsContext): DenseMatrix? = inverse(a, context.backend)

    /** То же, что [conditionInf], с реализацией линейной алгебры из [context]. */
    public fun conditionInf(
        a: DenseMatrix,
        context: NumericsContext,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate = conditionInf(a, tolerance, context.backend)

    /** То же, что [conditionEstimate], с реализацией линейной алгебры из [context]. */
    public fun conditionEstimate(
        a: DenseMatrix,
        context: NumericsContext,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate = conditionEstimate(a, tolerance, context.backend)

    /** То же, что [symmetricEigenvalues], с реализацией линейной алгебры из [context]. */
    public fun symmetricEigenvalues(a: DenseMatrix, context: NumericsContext): DoubleArray =
        symmetricEigenvalues(a, context.backend)

    /** То же, что [conditionSymmetric], с реализацией линейной алгебры из [context]. */
    public fun conditionSymmetric(a: DenseMatrix, context: NumericsContext): Double =
        conditionSymmetric(a, context.backend)
}
