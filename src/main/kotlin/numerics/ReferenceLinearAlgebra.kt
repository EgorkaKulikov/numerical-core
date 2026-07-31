package numerics

import kotlin.math.abs

/**
 * Эталонная (reference) реализация линейной алгебры на чистом Kotlin.
 *
 * Это «оракул корректности»: точная копия исходной ручной реализации
 * [LinearAlgebra], сохранённая для перекрёстной проверки оптимизированного
 * multik/OpenBLAS-бэкенда. Не используется в боевых вычислениях, только в тестах.
 *
 * НЕЗАВИСИМОСТЬ ОТ ПРОВЕРЯЕМОГО КОДА — главное свойство этого файла. Оракул не
 * ссылается ни на фасад `LinearAlgebra`, ни на `Backends`: иначе тест сверки
 * `LinearAlgebraVsReferenceTest` замкнулся бы сам на себя и перестал что-либо
 * доказывать. Общие с фасадом дешёвые операции берутся из [DenseOps] — третьего
 * объекта, который сам не зависит ни от фасада, ни от бэкендов.
 */
object ReferenceLinearAlgebra {

    /** Создаёт нулевую матрицу размера rows x cols. */
    private fun zeros(rows: Int, cols: Int): Array<DoubleArray> = DenseOps.zeros(rows, cols)

    /** Произведение матрицы A (m x k) на вектор x (k) -> вектор (m). */
    fun matVec(a: Array<DoubleArray>, x: DoubleArray): DoubleArray {
        val m = a.size
        val out = DoubleArray(m)
        for (i in 0 until m) {
            var s = 0.0
            val row = a[i]
            for (j in x.indices) s += row[j] * x[j]
            out[i] = s
        }
        return out
    }

    /** Транспонированное произведение A^T y, A: m x n, y: m -> вектор n. */
    fun matTransVec(a: Array<DoubleArray>, y: DoubleArray): DoubleArray {
        val m = a.size
        val n = a[0].size
        val out = DoubleArray(n)
        for (i in 0 until m) {
            val row = a[i]
            val yi = y[i]
            for (j in 0 until n) out[j] += row[j] * yi
        }
        return out
    }

    /** Произведение матриц A (m x k) на B (k x p) -> (m x p). */
    fun matMat(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val m = a.size
        val k = b.size
        val p = b[0].size
        val out = zeros(m, p)
        for (i in 0 until m) {
            val ai = a[i]
            val oi = out[i]
            for (l in 0 until k) {
                val ail = ai[l]
                if (ail == 0.0) continue
                val bl = b[l]
                for (j in 0 until p) oi[j] += ail * bl[j]
            }
        }
        return out
    }

    /** Произведение A^T diag(w) A для A: m x n, w: m -> симметричная n x n. */
    fun atWa(a: Array<DoubleArray>, w: DoubleArray): Array<DoubleArray> {
        val m = a.size
        val n = a[0].size
        val out = zeros(n, n)
        for (k in 0 until m) {
            val row = a[k]
            val wk = w[k]
            for (i in 0 until n) {
                val rwi = wk * row[i]
                if (rwi == 0.0) continue
                val outI = out[i]
                for (j in 0 until n) outI[j] += rwi * row[j]
            }
        }
        return out
    }

    /** Поэлементная сумма матриц A + s*B (одинаковые размеры). */
    fun addScaled(a: Array<DoubleArray>, b: Array<DoubleArray>, s: Double): Array<DoubleArray> {
        val out = Array(a.size) { a[it].copyOf() }
        for (i in a.indices) for (j in a[i].indices) out[i][j] += s * b[i][j]
        return out
    }

    /**
     * Порог вырожденности ведущего элемента, ОТНОСИТЕЛЬНЫЙ к масштабу матрицы.
     *
     * Используется как `PIVOT_RELATIVE_TOLERANCE * ||A||_inf`. Абсолютный порог
     * (ранее 1e-300) фактически проверял лишь строгий машинный ноль и пропускал
     * практически вырожденные матрицы: система с числом обусловленности ~1e18
     * решалась молча и возвращала мусор. Значение 1e-14 близко к машинному эпсилону
     * double (2.2e-16) с запасом на накопление ошибок исключения Гаусса.
     */
    private const val PIVOT_RELATIVE_TOLERANCE = 1e-14

    /**
     * Решение плотной СЛАУ A x = b методом LU с частичным выбором ведущего
     * элемента. Матрица A и вектор b не изменяются (работаем на копиях).
     *
     * О СООТНОШЕНИИ С ДРУГИМИ БЭКЕНДАМИ. Общий контракт вырожденности обеспечивает
     * ФАСАД [numerics.LinearAlgebra.solve] (проверка NaN/Inf и относительной невязки,
     * см. [numerics.LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE]), а не отдельные бэкенды.
     * Здешняя проверка ведущего элемента сохранена как СОБСТВЕННАЯ, БОЛЕЕ СТРОГАЯ
     * диагностика ручного LU: она видит сами ведущие элементы, чего нативный
     * LAPACK принципиально не показывает.
     *
     * Поэтому бэкенды НЕ являются «наблюдаемо неразличимыми» в строгом смысле
     * (прежняя редакция этого KDoc утверждала обратное — и это было ложно):
     *  - на ТОЧНО вырожденной матрице любого масштаба оба бэкенда бросают
     *    [IllegalStateException] — это гарантированное общее поведение;
     *  - на плохо ОБУСЛОВЛЕННОЙ, но невырожденной системе (cond ~ 1e16, например
     *    матрица Гильберта 12x12) реакции расходятся осмысленно: здешний LU может
     *    отказаться решать, а LAPACK — вернуть решение с малой невязкой; оба ответа
     *    честны, молчаливого мусора не возвращает ни один.
     *
     * ВАЖНО: этот метод остаётся НЕЗАВИСИМЫМ оракулом для тестов сверки бэкендов
     * и не вызывает ни фасад, ни активный бэкенд.
     *
     * @throws IllegalStateException при вырожденности или нечисловом результате.
     */
    fun solve(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
        val n = a.size
        val lu = Array(n) { a[it].copyOf() }
        val x = b.copyOf()
        // Масштаб матрицы: максимальная строчная сумма модулей (норма ||A||_inf).
        var matrixNorm = 0.0
        for (row in a) {
            var rowSum = 0.0
            for (v in row) rowSum += abs(v)
            matrixNorm = maxOf(matrixNorm, rowSum)
        }
        val pivotTolerance = PIVOT_RELATIVE_TOLERANCE * maxOf(matrixNorm, java.lang.Double.MIN_NORMAL)
        for (col in 0 until n) {
            var pivRow = col
            var pivVal = abs(lu[col][col])
            for (r in col + 1 until n) {
                val v = abs(lu[r][col])
                if (v > pivVal) { pivVal = v; pivRow = r }
            }
            if (pivVal <= pivotTolerance) error("LU: матрица вырождена (col=$col)")
            if (pivRow != col) {
                val t = lu[col]; lu[col] = lu[pivRow]; lu[pivRow] = t
                val tx = x[col]; x[col] = x[pivRow]; x[pivRow] = tx
            }
            val pivotR = lu[col]
            val pivot = pivotR[col]
            for (r in col + 1 until n) {
                val factor = lu[r][col] / pivot
                lu[r][col] = factor
                val rowR = lu[r]
                for (c in col + 1 until n) rowR[c] -= factor * pivotR[c]
                x[r] -= factor * x[col]
            }
        }
        // обратный ход
        for (i in n - 1 downTo 0) {
            var s = x[i]
            val row = lu[i]
            for (j in i + 1 until n) s -= row[j] * x[j]
            x[i] = s / row[i]
        }
        // Реакция на нечисловой результат: если во входных данных был NaN/Inf,
        // молча вернуть NaN-вектор нельзя. То же требование дублируется в фасаде (для
        // всех бэкендов), но здесь оно нужно и при ПРЯМОМ вызове оракула в тестах.
        for (v in x) {
            if (v.isNaN() || v.isInfinite()) error("LU: матрица вырождена (нечисловой результат)")
        }
        return x
    }
}

