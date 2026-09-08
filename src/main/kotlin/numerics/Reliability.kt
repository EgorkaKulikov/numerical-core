package numerics

import kotlin.math.abs

/**
 * Порог, ниже которого значение считается шумом округления, а не измерением.
 * Выбран на два–три порядка выше машинного эпсилона double (2.2e-16), чтобы отделить
 * погрешность метода от накопленной погрешности округления в суммах длины порядка 10²–10³.
 *
 * Величина сначала классифицируется ([measured]), и отношение ([ratio]) и порядок сходимости
 * ([orderOrNull], [reliableOrders]) по конструкции не вычисляются, если хотя бы один
 * аргумент лежит на уровне шума.
 */
const val MACHINE_NOISE_THRESHOLD: Double = 1e-13

/**
 * Величина вместе с суждением о её достоверности.
 *
 * Тип намеренно закрытый (`sealed`): пройти от числа к производной величине
 * (отношению, порядку сходимости) можно только через явное сопоставление
 * с образцом, поэтому недостоверное значение нельзя использовать по
 * невнимательности.
 */
sealed interface Measured {
    /** Само численное значение — доступно всегда, в том числе для печати с пометкой. */
    val value: Double

    /** Величина превышает порог достоверности и может участвовать в вычислениях. */
    data class Reliable(override val value: Double) : Measured

    /** Величина не превышает порог [threshold] (или не является конечной) — это шум. */
    data class AtNoiseLevel(override val value: Double, val threshold: Double) : Measured
}

/**
 * Классифицирует величину относительно порога достоверности.
 *
 * Нефинитные значения ([Double.NaN], бесконечности) считаются недостоверными:
 * они возникают при делении на нуль и при переполнении и не являются измерением.
 *
 * @throws IllegalArgumentException если `threshold <= 0` — нулевой порог означал
 *   бы отсутствие проверки, то есть ровно ту ошибку, которую тип предотвращает.
 */
fun measured(value: Double, threshold: Double = MACHINE_NOISE_THRESHOLD): Measured {
    require(threshold > 0.0) { "measured: требуется threshold > 0, получено $threshold" }
    return if (value.isFinite() && abs(value) >= threshold) {
        Measured.Reliable(value)
    } else {
        Measured.AtNoiseLevel(value, threshold)
    }
}

/**
 * Отношение двух величин или `null`, если хотя бы одна из них недостоверна.
 *
 * `null` здесь — не «не удалось посчитать», а «величина не определена»:
 * отношение чисел на уровне шума округления не несёт информации о методе.
 */
fun ratio(numerator: Measured, denominator: Measured): Double? =
    if (numerator is Measured.Reliable && denominator is Measured.Reliable) {
        numerator.value / denominator.value
    } else {
        null
    }

/**
 * Эмпирический порядок сходимости `log2(E_h / E_{h/2})` по паре достоверных
 * погрешностей или `null`.
 *
 * `null` возвращается, если хотя бы одна погрешность на уровне шума ([ratio])
 * либо их отношение неположительно (логарифм не определён).
 */
fun orderOrNull(coarse: Measured, fine: Measured): Double? {
    val r = ratio(coarse, fine) ?: return null
    return if (r <= 0.0) null else Math.log(r) / Math.log(2.0)
}

/**
 * Столбец порядков сходимости с применённым порогом достоверности —
 * вариант [orders], который отказывается считать порядок по шуму.
 *
 * СООТНОШЕНИЕ С [orders]. Обе функции дают для последней строки «не определено»
 * (там просто нет следующей строки). Отличие в том, что [orders] считает
 * порядок по любым положительным величинам, а эта функция — только по
 * величинам не ниже [threshold]; «не определено» здесь кодируется `null`, а не
 * [Double.NaN], чтобы значение нельзя было случайно подставить в арифметику.
 * Для печати в одном формате достаточно `reliableOrders(errs).map { it ?: Double.NaN }`,
 * и при всех погрешностях выше порога результат совпадает с `orders(errs)`.
 */
fun reliableOrders(errs: List<Double>, threshold: Double = MACHINE_NOISE_THRESHOLD): List<Double?> {
    val m = errs.map { measured(it, threshold) }
    return m.indices.map { i -> if (i + 1 < m.size) orderOrNull(m[i], m[i + 1]) else null }
}

/**
 * Константа `C_h = E_h / h^p` по достоверной погрешности или `null` —
 * вариант [constCh] с применённым порогом достоверности.
 *
 * @throws IllegalArgumentException если шаг `h` не положителен
 */
fun reliableConstCh(eh: Measured, h: Double, p: Double): Double? {
    require(h > 0) { "шаг сетки должен быть положительным, получено $h" }
    return if (eh is Measured.Reliable) constCh(eh.value, h, p) else null
}
