package numerics

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.log10
import kotlin.math.sqrt

// ============================================================================
// ОБУСЛОВЛЕННОСТЬ ПЛОТНЫХ МАТРИЦ
// ============================================================================

/**
 * Оценка числа обусловленности плотной матрицы — вместе с признаком её
 * достоверности.
 *
 * ЗАЧЕМ ОТДЕЛЬНЫЙ ТИП, А НЕ ПРОСТО `Double`. Число обусловленности само
 * вычисляется численно, и на почти вырожденных матрицах эта величина
 * недостоверна (см. [Conditioning.conditionInf]). Возвращать одно число
 * означало бы молча выдавать шум: вызывающий не может отличить `cond = 2.25`
 * от `cond = 3.7e18`, где вторая цифра — артефакт округления. Поэтому оценка
 * всегда сопровождается [inversionResidual] — измеренной невязкой обращения
 * `‖A A⁻¹ − I‖∞`, по которой достоверность проверяется вызывающим или методом
 * [isReliable].
 *
 * @property condInf оценка `cond∞(A) = ‖A‖∞ · ‖A⁻¹‖∞`; [Double.POSITIVE_INFINITY],
 *   если обращение не удалось (решатель признал матрицу вырожденной).
 * @property inversionResidual невязка обращения `‖A A⁻¹ − I‖∞`;
 *   [Double.POSITIVE_INFINITY], если обращение не удалось.
 * @property tolerance порог, по которому оценка признаётся достоверной.
 */
data class ConditionEstimate(
    val condInf: Double,
    val inversionResidual: Double,
    val tolerance: Double,
) {
    /** Достоверна ли оценка: обращение состоялось и его невязка не превышает [tolerance]. */
    val isReliable: Boolean
        get() = condInf.isFinite() && inversionResidual <= tolerance

    /** [condInf], если оценка достоверна, иначе `null` — чтобы недостоверное число нельзя было использовать по невнимательности. */
    fun valueOrNull(): Double? = if (isReliable) condInf else null
}

/**
 * Оценка ОТНОСИТЕЛЬНОЙ ПРЯМОЙ ошибки решения СЛАУ — вместе с суждением о том,
 * существует ли она вообще и можно ли ей верить.
 *
 * ОБРАТНАЯ И ПРЯМАЯ ОШИБКА — РАЗНЫЕ ВЕЛИЧИНЫ. Обратная ошибка отвечает на вопрос
 * «какую систему алгоритм решил точно»: `ω = ‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)`.
 * У LU с частичным выбором она мала ВСЕГДА, в том числе на численно вырожденной
 * матрице (измерено 3.97e-16); ровно её контролирует постпроверка решения по
 * порогу [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE], и по построению та ловит
 * только откровенный мусор. Прямая ошибка отвечает на ДРУГОЙ вопрос — насколько
 * возвращённый `x` отличается от истинного решения `x*` — и ограничивается
 * произведением `‖x − x*‖∞ / ‖x*‖∞ <= cond∞(A) · ω`. При `cond∞ ~ 2.6e10` и
 * `ω ~ 1e-16` верных десятичных разрядов остаётся около шести, а при
 * `cond∞ ~ 1/ε ~ 4.5e15` — ни одного. МАЛАЯ ОБРАТНАЯ ОШИБКА НЕ ОЗНАЧАЕТ, ЧТО
 * РЕЗУЛЬТАТ ТОЧЕН.
 *
 * ЗАЧЕМ ЗАКРЫТЫЙ ТИП, А НЕ `Double`. Множитель `cond∞(A)` сам вычисляется
 * численно и на почти вырожденных матрицах недостоверен (граница применимости —
 * в KDoc [Conditioning]). Вернуть в этом случае число значило бы выдать шум за
 * оценку. Поэтому три режима разделены КОНСТРУКТОРАМИ, и получить из них число
 * можно только явным сопоставлением с образцом либо через [relativeBoundOrNull],
 * который возвращает `null` там, где числа нет.
 */
sealed interface ForwardError {
    /**
     * Измеренная относительная ОБРАТНАЯ ошибка — доступна во всех режимах: она
     * именно измеряется (см. [Conditioning.relativeBackwardError]), а не
     * оценивается, и потому достоверна всегда.
     */
    val backwardError: Double

    /**
     * Оценка существует и достоверна: `‖x − x*‖∞ / ‖x*‖∞ <= [relativeBound]`.
     *
     * @property cond использованное число обусловленности.
     * @property relativeBound произведение `cond · backwardError`.
     */
    data class Bounded(
        override val backwardError: Double,
        val cond: Double,
        val relativeBound: Double,
    ) : ForwardError

    /**
     * Конечной границы НЕТ ВОВСЕ: матрица численно вырождена — спектральная
     * оценка даёт `σ_min = 0` либо обращение не удалось.
     *
     * Содержательно отличается от [Unreliable]: там оценка `cond` существует, но
     * ей нельзя верить, а здесь конечного числа обусловленности у матрицы нет.
     */
    data class NoFiniteBound(override val backwardError: Double) : ForwardError

    /**
     * Оценка `cond` конечна, но недостоверна ([ConditionEstimate.isReliable]
     * равно `false`), поэтому границы прямой ошибки не существует как измерения.
     *
     * @property condition сама недостоверная оценка — для печати с пометкой и
     *   для разбора причины: невязка обращения лежит в ней же.
     */
    data class Unreliable(
        override val backwardError: Double,
        val condition: ConditionEstimate,
    ) : ForwardError

    /** Граница прямой ошибки или `null` в режимах [NoFiniteBound] и [Unreliable]. */
    fun relativeBoundOrNull(): Double? = (this as? Bounded)?.relativeBound

    /**
     * Сколько ДЕСЯТИЧНЫХ разрядов результата заведомо уцелело: `-log10(границы)`,
     * обрезанное снизу нулём (граница `>= 1` означает «ни одного гарантированно
     * верного разряда») и сверху 16 (полная мантисса double).
     *
     * `null` — там же, где `null` у [relativeBoundOrNull]: нет границы — нет и
     * числа разрядов.
     */
    fun survivingDigitsOrNull(): Double? {
        val bound = relativeBoundOrNull() ?: return null
        if (bound <= 0.0) return 16.0
        return minOf(16.0, maxOf(0.0, -log10(bound)))
    }
}

/**
 * Числа обусловленности плотных матриц: оценка через явное обращение и
 * спектральная оценка для симметричных матриц.
 *
 * ЗАЧЕМ. Каждый решатель второго рода собирает плотную матрицу `I − λK_h` и
 * решает с ней СЛАУ. Плотный решатель обратно устойчив, но на численно
 * вырожденной матрице возвращает мусор БЕЗ ПРЕДУПРЕЖДЕНИЯ: алгебраически
 * нейтральные перезаписи выражений при `cond∞ ≈ 2.6e10` меняли результат на
 * 15 %. Возможность узнать обусловленность собранной матрицы — необходимое
 * условие для честного протокола воспроизводимости.
 *
 * ГРАНИЦА ПРИМЕНИМОСТИ [conditionInf] (установлено экспериментально).
 *  - НА ХОРОШО ОБУСЛОВЛЕННЫХ МАТРИЦАХ ДОСТОВЕРНО. Для системы второго рода
 *    модельной задачи M1 оценка через обращение совпала с независимой
 *    спектральной оценкой методом вращений Якоби на 6–7 значащих цифр:
 *    2.252006 / 2.185581 / 2.163922 при `n = 8 / 32 / 128`.
 *  - НА ПОЧТИ ВЫРОЖДЕННЫХ МАТРИЦАХ НЕДОСТОВЕРНО. При параметре регуляризации
 *    `α <= 1e-12` невязка обращения `‖A A⁻¹ − I‖∞` достигала 0.08…760, а сами
 *    значения `cond` оказывались немонотонным шумом порядка 1e16…1e19. Для
 *    матрицы дискретизации ядра `1/(1 + t + s)` независимый метод Якоби даёт
 *    `σ_min` РОВНО 0 — то есть никакого конечного числа обусловленности у неё
 *    нет вовсе, и любое напечатанное значение было бы вымыслом.
 *  - ПОЧЕМУ ТАК. `A⁻¹` вычисляется решением `n` систем `A x = e_j` тем же
 *    плотным решателем. Его обратная ошибка мала (измерено 3.97e-16), но прямая
 *    ошибка растёт как `cond(A) · ε`, поэтому при `cond ~ 1/ε ~ 4.5e15` в
 *    столбцах `A⁻¹` не остаётся ни одной верной цифры. Невязка `‖A A⁻¹ − I‖∞`
 *    и есть прямой индикатор этой потери.
 *  - ДЛЯ СИММЕТРИЧНЫХ МАТРИЦ надёжнее [smallestMagnitudeEigenvalue] /
 *    [conditionSymmetric]: метод вращений Якоби работает ортогональными
 *    преобразованиями, не обращает матрицу и на вырожденной матрице честно
 *    выдаёт нулевое собственное значение вместо шума.
 */
object Conditioning {

    /**
     * Порог достоверности оценки [conditionInf] по невязке обращения `‖A A⁻¹ − I‖∞`.
     *
     * ОТКУДА ЗНАЧЕНИЕ 1e-8. На хорошо обусловленных матрицах невязка держится на
     * уровне 1e-16…1e-14 (машинный эпсилон, умноженный на размер), на почти
     * вырожденных — измерено 0.08…760. Между этими режимами четыре порядка
     * пустого места; 1e-8 лежит примерно посередине в логарифмической шкале и
     * соответствует потере половины значащих разрядов: при большей невязке в
     * оценке `cond` не остаётся и половины верных цифр, и печатать её нельзя.
     */
    const val INVERSION_RESIDUAL_TOLERANCE: Double = 1e-8

    /** Строчная (равномерная) норма матрицы: `‖A‖∞ = max_i Σ_j |a_ij|`. */
    fun matrixNormInf(a: Array<DoubleArray>): Double {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "matrixNormInf: пустая матрица A" }
        var m = 0.0
        for (row in a) {
            var s = 0.0
            for (v in row) s += abs(v)
            m = maxOf(m, s)
        }
        return m
    }

    /**
     * Обращение квадратной матрицы столбец за столбцом: `A x_j = e_j`.
     *
     * @return `A⁻¹` или `null`, если решатель признал матрицу вырожденной либо
     *   вернул нефинитные компоненты: часть бэкендов не бросает исключение, а
     *   молча выдаёт NaN/Inf, и такой ответ обязан стать `null`, а не мусором.
     *   Достоверность конечного результата НЕ гарантируется — её измеряет
     *   [inversionResidual]; см. границу применимости в KDoc [Conditioning].
     */
    fun inverse(a: Array<DoubleArray>): Array<DoubleArray>? {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "inverse: пустая матрица A" }
        require(a.size == a[0].size) { "inverse: требуется квадратная матрица, получено ${a.size}x${a[0].size}" }
        val n = a.size
        val inv = LinearAlgebra.zeros(n, n)
        for (j in 0 until n) {
            val e = DoubleArray(n)
            e[j] = 1.0
            val x = try {
                LinearAlgebra.solve(a, e)
            } catch (_: RuntimeException) {
                return null
            }
            for (i in 0 until n) {
                if (!x[i].isFinite()) return null
                inv[i][j] = x[i]
            }
        }
        return inv
    }

    /** Невязка обращения `‖A B − I‖∞` — прямая мера того, сколько разрядов уцелело в `B ≈ A⁻¹`. */
    fun inversionResidual(a: Array<DoubleArray>, inv: Array<DoubleArray>): Double {
        val n = a.size
        val prod = LinearAlgebra.matMat(a, inv)
        for (i in 0 until n) prod[i][i] -= 1.0
        return matrixNormInf(prod)
    }

    /**
     * Оценка `cond∞(A) = ‖A‖∞ · ‖A⁻¹‖∞` через явное обращение, вместе с
     * невязкой обращения.
     *
     * Метод НЕ бросает исключение на вырожденной матрице и НЕ возвращает
     * молча мусор: недостоверный результат помечается через
     * [ConditionEstimate.isReliable]. Границу применимости и происхождение
     * порога [INVERSION_RESIDUAL_TOLERANCE] см. в KDoc [Conditioning].
     *
     * @param tolerance порог достоверности по невязке обращения.
     */
    fun conditionInf(
        a: Array<DoubleArray>,
        tolerance: Double = INVERSION_RESIDUAL_TOLERANCE,
    ): ConditionEstimate {
        require(tolerance > 0.0) { "conditionInf: требуется tolerance > 0, получено $tolerance" }
        val inv = inverse(a)
            ?: return ConditionEstimate(Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY, tolerance)
        val cond = matrixNormInf(a) * matrixNormInf(inv)
        return ConditionEstimate(cond, inversionResidual(a, inv), tolerance)
    }

    /**
     * Собственные значения симметричной матрицы методом циклических вращений
     * Якоби, по возрастанию.
     *
     * ЗАЧЕМ ИМЕННО ЯКОБИ. Метод работает только ортогональными преобразованиями:
     * матрица не обращается, спектр не искажается делением на почти нулевые
     * величины. На численно вырожденных матрицах он выдаёт собственное значение,
     * равное нулю, а не шум порядка 1e16, — поэтому для симметричных матриц он
     * служит независимой проверкой оценки [conditionInf].
     *
     * @param maxSweeps максимальное число проходов по всем внедиагональным парам.
     * @throws IllegalArgumentException если матрица не квадратная или её асимметрия
     *   превышает `1e-12 * ‖A‖∞`.
     */
    fun symmetricEigenvalues(a: Array<DoubleArray>, maxSweeps: Int = 100): DoubleArray {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "symmetricEigenvalues: пустая матрица A" }
        require(a.size == a[0].size) { "symmetricEigenvalues: требуется квадратная матрица, получено ${a.size}x${a[0].size}" }
        require(maxSweeps >= 1) { "symmetricEigenvalues: требуется maxSweeps >= 1, получено $maxSweeps" }
        val scale = matrixNormInf(a)
        require(LinearAlgebra.maxAsymmetry(a) <= 1e-12 * scale) {
            "symmetricEigenvalues: матрица не симметрична (max|A - A^T| = ${LinearAlgebra.maxAsymmetry(a)})"
        }
        val n = a.size
        val m = Array(n) { i -> a[i].copyOf() }
        // Порог сходимости: внедиагональная часть неотличима от шума округления.
        val eps = 1e-15 * maxOf(scale, Double.MIN_VALUE)
        for (sweep in 0 until maxSweeps) {
            var off = 0.0
            for (p in 0 until n) for (q in p + 1 until n) off = maxOf(off, abs(m[p][q]))
            if (off <= eps) break
            for (p in 0 until n) for (q in p + 1 until n) {
                if (abs(m[p][q]) <= eps) continue
                // Классическое вращение: theta = (a_qq - a_pp) / (2 a_pq), t = sign/(|theta| + sqrt(theta^2 + 1)).
                val theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q])
                val t = (if (theta >= 0.0) 1.0 else -1.0) / (abs(theta) + hypot(theta, 1.0))
                val c = 1.0 / sqrt(t * t + 1.0)
                val s = t * c
                for (k in 0 until n) {
                    val akp = m[k][p]
                    val akq = m[k][q]
                    m[k][p] = c * akp - s * akq
                    m[k][q] = s * akp + c * akq
                }
                for (k in 0 until n) {
                    val apk = m[p][k]
                    val aqk = m[q][k]
                    m[p][k] = c * apk - s * aqk
                    m[q][k] = s * apk + c * aqk
                }
            }
        }
        val eig = DoubleArray(n) { i -> m[i][i] }
        eig.sort()
        return eig
    }

    /**
     * Наименьшее по модулю собственное значение симметричной матрицы —
     * её `σ_min` (для симметричной матрицы сингулярные числа суть `|λ_i|`).
     *
     * Ровно `0.0` означает численную вырожденность: конечного числа
     * обусловленности у такой матрицы нет, и печатать его нельзя.
     */
    fun smallestMagnitudeEigenvalue(a: Array<DoubleArray>, maxSweeps: Int = 100): Double {
        val eig = symmetricEigenvalues(a, maxSweeps)
        var m = Double.POSITIVE_INFINITY
        for (v in eig) m = minOf(m, abs(v))
        return m
    }

    /**
     * Спектральное число обусловленности симметричной матрицы
     * `cond₂(A) = max|λ_i| / min|λ_i|`.
     *
     * Возвращает [Double.POSITIVE_INFINITY] при `min|λ_i| = 0` — честный ответ
     * «матрица вырождена», а не большое случайное число.
     */
    fun conditionSymmetric(a: Array<DoubleArray>, maxSweeps: Int = 100): Double {
        val eig = symmetricEigenvalues(a, maxSweeps)
        var lo = Double.POSITIVE_INFINITY
        var hi = 0.0
        for (v in eig) {
            lo = minOf(lo, abs(v))
            hi = maxOf(hi, abs(v))
        }
        return if (lo == 0.0) Double.POSITIVE_INFINITY else hi / lo
    }

    // --- Прямая ошибка решения СЛАУ -----------------------------------------

    /**
     * Относительная ОБРАТНАЯ ошибка решения `A x = b`:
     * `‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)`.
     *
     * Это РОВНО та величина, которую постпроверка [LinearAlgebra.solve] сравнивает
     * с порогом [LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE]: нормировка
     * `max(‖A‖∞‖x‖∞, ‖b‖∞)` взята оттуда дословно — иначе в проекте появилось бы
     * две разные «обратные ошибки» с расходящимися значениями. Обоснование самой
     * нормировки (в частности, зачем в ней `‖b‖∞`) — в KDoc этого порога.
     *
     * Величина достоверна всегда: она ИЗМЕРЯЕТСЯ одним проходом по `A`, а не
     * оценивается. Прямую (то есть интересующую пользователя) ошибку она НЕ
     * ограничивает: множителем служит `cond∞(A)`, см. [ForwardError].
     *
     * Нулевой масштаб (`A = 0`, `b = 0`) даёт `0.0`: невязка там тождественно
     * нулевая, и деление ноля на ноль здесь дало бы `NaN` вместо честного
     * «ошибки нет».
     */
    fun relativeBackwardError(a: Array<DoubleArray>, b: DoubleArray, x: DoubleArray): Double {
        require(a.isNotEmpty() && a[0].isNotEmpty()) { "relativeBackwardError: пустая матрица A" }
        require(a.size == a[0].size) {
            "relativeBackwardError: требуется квадратная A, получено ${a.size}x${a[0].size}"
        }
        require(a.size == b.size) {
            "relativeBackwardError: несогласованные размеры A(${a.size} строк) и b(${b.size})"
        }
        require(a.size == x.size) {
            "relativeBackwardError: несогласованные размеры A(${a.size} строк) и x(${x.size})"
        }
        val n = a.size
        var matrixNorm = 0.0 // ‖A‖∞
        var residual = 0.0 // ‖Ax − b‖∞
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
        val scale = maxOf(matrixNorm * LinearAlgebra.normInf(x), LinearAlgebra.normInf(b))
        return if (scale == 0.0) 0.0 else residual / scale
    }

    /**
     * Граница относительной ПРЯМОЙ ошибки `cond∞(A) · ω` по готовой оценке
     * обусловленности и измеренной обратной ошибке `ω`.
     *
     * Функция ЧИСТАЯ и ничего не вычисляет заново: она лишь переводит пару
     * «оценка `cond` + измеренная обратная ошибка» в один из трёх режимов
     * [ForwardError], не позволяя недостоверному `cond` превратиться в число.
     * Смысл режимов описан в KDoc [ForwardError].
     *
     * @param backwardError результат [relativeBackwardError]; требуется конечным и неотрицательным.
     */
    fun forwardError(condition: ConditionEstimate, backwardError: Double): ForwardError {
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
     * Граница относительной ПРЯМОЙ ошибки для СИММЕТРИЧНОЙ матрицы — через
     * спектральную оценку [conditionSymmetric], а не через обращение.
     *
     * ЗАЧЕМ ОТДЕЛЬНЫЙ ПУТЬ. Оценка через обращение на почти вырожденной матрице
     * сама недостоверна и даёт режим [ForwardError.Unreliable] — «числа нет,
     * причина неизвестна». Метод вращений Якоби матрицу не обращает и на
     * численно вырожденной матрице честно выдаёт `σ_min = 0`, поэтому здесь
     * различаются два разных факта: «конечного числа обусловленности нет вовсе»
     * ([ForwardError.NoFiniteBound]) и «`cond` велико, но конечно»
     * ([ForwardError.Bounded] с большой границей). Именно так себя ведёт матрица
     * дискретизации ядра `1/(1 + t + s)`: `σ_min` у неё РОВНО 0, тогда как оценка
     * через обращение выдавала немонотонный шум порядка 1e16…1e19.
     *
     * Режим [ForwardError.Unreliable] эта функция не возвращает никогда:
     * ортогональные преобразования не теряют достоверность так, как её теряет
     * обращение.
     *
     * ЦЕНА: O(maxSweeps · n³) — дороже обращения, поэтому путь применяется
     * осознанно и только там, где симметрия действительно есть.
     *
     * @throws IllegalArgumentException если матрица не симметрична (проверка
     *   наследуется от [symmetricEigenvalues]).
     */
    fun forwardErrorSymmetric(
        a: Array<DoubleArray>,
        backwardError: Double,
        maxSweeps: Int = 100,
    ): ForwardError {
        require(backwardError.isFinite() && backwardError >= 0.0) {
            "forwardErrorSymmetric: обратная ошибка должна быть конечной и неотрицательной, получено $backwardError"
        }
        val cond = conditionSymmetric(a, maxSweeps)
        return if (!cond.isFinite()) {
            ForwardError.NoFiniteBound(backwardError)
        } else {
            ForwardError.Bounded(backwardError, cond, cond * backwardError)
        }
    }
}
