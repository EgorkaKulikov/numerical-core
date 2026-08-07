package numerics

import kotlin.math.abs
import kotlin.math.hypot
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
}
