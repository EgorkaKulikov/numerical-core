package numerics.backend

import numerics.DenseMatrix

/**
 * Интерфейс реализации плотной линейной алгебры над матрицами [DenseMatrix]
 * (плоский столбцовый формат) и плоскими векторами [DoubleArray].
 *
 * Интерфейс подключаемой реализации: через него единая точка входа [numerics.LinearAlgebra]
 * обращается к вычислительному ядру; реализация подбирается в [Backends] и передаётся
 * явным параметром.
 * Столбцовый формат матриц совпадает с соглашением BLAS/LAPACK, поэтому данные
 * уходят в вычислительное ядро без перекладки.
 *
 * Контракт реализаций:
 *  - входные матрицы и векторы не изменяются; там, где ядро пишет результат на место
 *    входа, реализация работает с копией;
 *  - согласованность размерностей проверяет вызывающий код ([numerics.LinearAlgebra]),
 *    реализация вправе полагаться на неё и не обязана дублировать проверки;
 *  - результат всегда новый объект, не разделяющий память с входом.
 */
public interface LinAlgBackend {

    /** Человекочитаемое имя реализации для журнала и диагностики. */
    public val name: String

    /** Истина, если вычисления выполняет системная (нативная) библиотека BLAS/LAPACK. */
    public val isNative: Boolean

    /**
     * Обновление вектора `y += alpha·x` на месте (daxpy).
     * @throws IllegalArgumentException если длины `x` и `y` различаются.
     */
    public fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray)

    /** Произведение A·x для A размера m×k и x длины k; результат длины m. */
    public fun matVec(a: DenseMatrix, x: DoubleArray): DoubleArray

    /** Произведение Aᵀ·y для A размера m×n и y длины m; результат длины n. */
    public fun matTransVec(a: DenseMatrix, y: DoubleArray): DoubleArray

    /** Произведение A·B для A размера m×k и B размера k×p; результат m×p. */
    public fun matMat(a: DenseMatrix, b: DenseMatrix): DenseMatrix

    /** Произведение Aᵀ·B для A размера m×n и B размера m×p; результат n×p. */
    public fun matTransMat(a: DenseMatrix, b: DenseMatrix): DenseMatrix

    /**
     * Решение системы A·X = B для квадратной A и правых частей — столбцов B.
     * Разложение LU с частичным выбором ведущего элемента.
     *
     * @return матрица X тех же размеров, что B.
     * @throws IllegalStateException если в ходе разложения встретился нулевой ведущий
     *   элемент (матрица вырождена).
     */
    public fun solve(a: DenseMatrix, b: DenseMatrix): DenseMatrix

    /**
     * Разложение LU с частичным выбором ведущего элемента: P·A = L·U.
     * Обнаруженная вырожденность не прерывает разложение и отражается в
     * [LuFactorization.singularAt].
     */
    public fun luFactor(a: DenseMatrix): LuFactorization

    /**
     * Решение A·X = B по готовому разложению [lu] квадратной матрицы A.
     * @throws IllegalArgumentException если разложение вырождено ([LuFactorization.isSingular]).
     */
    public fun luSolve(lu: LuFactorization, b: DenseMatrix): DenseMatrix

    /**
     * Обратная матрица как решение A·X = I по LU-разложению; невязка ‖A·X − I‖ контролируется
     * так же, как при решении системы. Возвращает `null`, если A вырождена.
     */
    public fun inverse(a: DenseMatrix): DenseMatrix?

    /**
     * Разложение Холецкого A = L·Lᵀ для симметричной положительно определённой A;
     * читается только нижний треугольник A.
     * @return нижнетреугольная L (элементы выше диагонали равны нулю) либо `null`,
     *   если A не является положительно определённой.
     */
    public fun cholesky(a: DenseMatrix): DenseMatrix?

    /**
     * Собственные значения симметричной матрицы по возрастанию; читается только
     * верхний треугольник A.
     * @throws IllegalStateException если итерационный алгоритм не сошёлся.
     */
    public fun symmetricEigenvalues(a: DenseMatrix): DoubleArray

    /**
     * Оценка обратного числа обусловленности в норме «1»: приближение к
     * `1 / (‖A‖₁ · ‖A⁻¹‖₁)` по разложению [lu] и известной норме [norm1] исходной матрицы.
     * Значение 0 означает численно вырожденную матрицу.
     */
    public fun reciprocalCondition1(lu: LuFactorization, norm1: Double): Double

    /** Матричная норма указанного вида; для пустой матрицы — 0. */
    public fun norm(a: DenseMatrix, kind: MatrixNorm): Double
}

/** Виды матричных норм, доступные в [LinAlgBackend.norm]. */
public enum class MatrixNorm {
    /** Наибольшая сумма модулей по столбцам, ‖A‖₁. */
    ONE,

    /** Наибольшая сумма модулей по строкам, ‖A‖∞. */
    INF,

    /** Корень из суммы квадратов всех элементов, ‖A‖F. */
    FROBENIUS,

    /** Наибольший модуль элемента, max|aᵢⱼ| (не является матричной нормой в строгом смысле). */
    MAX,
}

/**
 * Результат разложения LU с частичным выбором ведущего элемента.
 *
 * @property lu множители L и U в одной матрице: единичная нижняя треугольная L под
 *   диагональю, верхняя треугольная U на диагонали и выше.
 * @property ipiv перестановки строк: строка i менялась со строкой `ipiv[i]` (нумерация с 1).
 * @property singularAt 0, если матрица невырождена; иначе номер k (с 1) первого нулевого
 *   диагонального элемента U — разложение завершено, но решать систему по нему нельзя.
 */
public class LuFactorization(public val lu: DenseMatrix, public val ipiv: IntArray, public val singularAt: Int) {
    /** Истина, если в ходе разложения встретился нулевой ведущий элемент. */
    public val isSingular: Boolean get() = singularAt > 0
}
