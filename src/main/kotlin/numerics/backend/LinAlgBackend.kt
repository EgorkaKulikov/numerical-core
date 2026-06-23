package numerics.backend

/**
 * Подключаемый бэкенд линейной алгебры (SPI, Service Provider Interface).
 *
 * Это **точка расширения** для высокопроизводительных реализаций тяжёлых
 * операций над плотными матрицами [Array]<[DoubleArray]> и векторами
 * [DoubleArray]. Чтобы добавить новый движок (например, GPU/CUDA, Metal,
 * распределённый BLAS), достаточно:
 *  1. реализовать этот интерфейс;
 *  2. зарегистрировать реализацию в [Backends].
 *
 * Никакой код решателей и фасад [numerics.LinearAlgebra] при этом не меняется:
 * фасад делегирует операции активному бэкенду [Backends.active].
 *
 * В интерфейс вынесены только «тяжёлые» операции, выигрывающие от аппаратного
 * ускорения (умножения и решение СЛАУ). Дешёвые скалярные/служебные операции
 * (zeros, identity, norm2, normInf, cholesky, maxAsymmetry) остаются прямо в
 * [numerics.LinearAlgebra] и не входят в SPI.
 *
 * Контракт реализаций:
 *  - входные массивы (матрицы и векторы) **не должны мутироваться**;
 *  - [solve] обязан бросать [IllegalStateException] на вырожденном входе
 *    (в т.ч. если нативный решатель вернул NaN/Inf вместо исключения);
 *  - семантика результатов идентична эталону [numerics.ReferenceLinearAlgebra]
 *    (совпадение на хорошо обусловленных входах в пределах ~1e-9).
 */
interface LinAlgBackend {

    /** Человекочитаемое имя бэкенда (для логов/бенчмарков/диагностики). */
    val name: String

    /**
     * Может ли этот бэкенд работать на текущей машине/рантайме?
     *
     * Проверка должна быть реальной (например, пробный вызов нативной
     * библиотеки), а не предполагаемой: на основании этого флага [Backends]
     * выполняет автоматический выбор и откат на доступный бэкенд.
     */
    fun isAvailable(): Boolean

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray>

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray>

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray>

    /**
     * Решение плотной СЛАУ A x = b. Входные A и b не изменяются.
     * @throws IllegalStateException при вырожденности.
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray
}
