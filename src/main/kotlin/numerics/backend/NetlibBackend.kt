package numerics.backend

import dev.ludovic.netlib.blas.BLAS
import dev.ludovic.netlib.lapack.LAPACK
import dev.ludovic.netlib.lapack.NativeLAPACK
import numerics.DenseMatrix
import org.netlib.util.doubleW
import org.netlib.util.intW

/**
 * Реализация [LinAlgBackend] поверх BLAS/LAPACK из библиотеки netlib.
 *
 * Пара [blas]/[lapack] задаёт вычислительное ядро: системная библиотека через JNI
 * либо переносимая реализация на Java; интерфейс вызовов у них общий, поэтому один
 * класс обслуживает оба варианта. Матрицы [DenseMatrix] хранятся в столбцовом порядке,
 * и их массивы передаются в ядро напрямую; где LAPACK пишет результат на место входа,
 * предварительно снимается копия — входы никогда не изменяются.
 */
class NetlibBackend internal constructor(
    private val blas: BLAS,
    private val lapack: LAPACK,
) : LinAlgBackend {

    override val isNative: Boolean = lapack is NativeLAPACK

    override val name: String =
        if (isNative) "netlib ${lapack.javaClass.simpleName} (нативная BLAS/LAPACK системы)"
        else "netlib ${lapack.javaClass.simpleName} (Java)"

    override fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray) {
        require(x.size == y.size) { "axpy: длины векторов должны совпадать, получено ${x.size} и ${y.size}" }
        if (x.isEmpty()) return
        blas.daxpy(x.size, alpha, x, 1, y, 1)
    }

    override fun matVec(a: DenseMatrix, x: DoubleArray): DoubleArray {
        val y = DoubleArray(a.rows)
        if (a.rows == 0 || a.cols == 0) return y
        blas.dgemv("N", a.rows, a.cols, 1.0, a.data, a.rows, x, 1, 0.0, y, 1)
        return y
    }

    override fun matTransVec(a: DenseMatrix, y: DoubleArray): DoubleArray {
        val out = DoubleArray(a.cols)
        if (a.rows == 0 || a.cols == 0) return out
        blas.dgemv("T", a.rows, a.cols, 1.0, a.data, a.rows, y, 1, 0.0, out, 1)
        return out
    }

    override fun matMat(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix.zeros(a.rows, b.cols)
        if (a.rows == 0 || b.cols == 0 || a.cols == 0) return c
        blas.dgemm("N", "N", a.rows, b.cols, a.cols, 1.0, a.data, a.rows, b.data, b.rows, 0.0, c.data, a.rows)
        return c
    }

    override fun matTransMat(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val c = DenseMatrix.zeros(a.cols, b.cols)
        if (a.cols == 0 || b.cols == 0 || a.rows == 0) return c
        blas.dgemm("T", "N", a.cols, b.cols, a.rows, 1.0, a.data, a.rows, b.data, b.rows, 0.0, c.data, a.cols)
        return c
    }

    override fun solve(a: DenseMatrix, b: DenseMatrix): DenseMatrix {
        val n = a.rows
        val nrhs = b.cols
        val aCopy = a.data.copyOf()
        val bCopy = b.data.copyOf()
        val ipiv = IntArray(n)
        val info = intW(0)
        lapack.dgesv(n, nrhs, aCopy, maxOf(1, n), ipiv, bCopy, maxOf(1, n), info)
        checkArgs("dgesv", info)
        if (info.`val` > 0) {
            throw IllegalStateException("матрица вырождена: нулевой ведущий элемент в позиции ${info.`val`}")
        }
        return DenseMatrix.fromColumnMajor(n, nrhs, bCopy)
    }

    override fun luFactor(a: DenseMatrix): LuFactorization {
        val lu = a.data.copyOf()
        val ipiv = IntArray(minOf(a.rows, a.cols))
        val info = intW(0)
        lapack.dgetrf(a.rows, a.cols, lu, maxOf(1, a.rows), ipiv, info)
        checkArgs("dgetrf", info)
        return LuFactorization(DenseMatrix.fromColumnMajor(a.rows, a.cols, lu), ipiv, info.`val`)
    }

    override fun luSolve(lu: LuFactorization, b: DenseMatrix): DenseMatrix {
        require(!lu.isSingular) { "разложение LU вырождено: нулевой ведущий элемент в позиции ${lu.singularAt}" }
        val n = lu.lu.rows
        val bCopy = b.data.copyOf()
        val info = intW(0)
        lapack.dgetrs("N", n, b.cols, lu.lu.data, maxOf(1, n), lu.ipiv, bCopy, maxOf(1, n), info)
        checkArgs("dgetrs", info)
        return DenseMatrix.fromColumnMajor(n, b.cols, bCopy)
    }

    override fun inverse(a: DenseMatrix): DenseMatrix? {
        val n = a.rows
        val inv = a.data.copyOf()
        val ipiv = IntArray(n)
        val info = intW(0)
        lapack.dgetrf(n, n, inv, maxOf(1, n), ipiv, info)
        checkArgs("dgetrf", info)
        if (info.`val` > 0) return null
        val query = DoubleArray(1)
        lapack.dgetri(n, inv, maxOf(1, n), ipiv, query, -1, info)
        checkArgs("dgetri", info)
        val lwork = maxOf(query[0].toInt(), n, 1)
        lapack.dgetri(n, inv, maxOf(1, n), ipiv, DoubleArray(lwork), lwork, info)
        checkArgs("dgetri", info)
        if (info.`val` > 0) return null
        return DenseMatrix.fromColumnMajor(n, n, inv)
    }

    override fun cholesky(a: DenseMatrix): DenseMatrix? {
        val n = a.rows
        val l = a.data.copyOf()
        val info = intW(0)
        lapack.dpotrf("L", n, l, maxOf(1, n), info)
        checkArgs("dpotrf", info)
        if (info.`val` > 0) return null
        for (j in 1 until n) for (i in 0 until j) l[i + j * n] = 0.0
        return DenseMatrix.fromColumnMajor(n, n, l)
    }

    override fun symmetricEigenvalues(a: DenseMatrix): DoubleArray {
        val n = a.rows
        if (n == 0) return DoubleArray(0)
        val work = a.data.copyOf()
        val w = DoubleArray(n)
        val info = intW(0)
        val query = DoubleArray(1)
        lapack.dsyev("N", "U", n, work, n, w, query, -1, info)
        checkArgs("dsyev", info)
        val lwork = maxOf(query[0].toInt(), 3 * n - 1, 1)
        lapack.dsyev("N", "U", n, work, n, w, DoubleArray(lwork), lwork, info)
        checkArgs("dsyev", info)
        if (info.`val` > 0) throw IllegalStateException("собственные значения не сошлись")
        return w
    }

    override fun reciprocalCondition1(lu: LuFactorization, norm1: Double): Double {
        val n = lu.lu.rows
        if (n == 0) return 1.0
        val rcond = doubleW(0.0)
        val info = intW(0)
        lapack.dgecon("1", n, lu.lu.data, n, norm1, rcond, DoubleArray(4 * n), IntArray(n), info)
        checkArgs("dgecon", info)
        return rcond.`val`
    }

    override fun norm(a: DenseMatrix, kind: MatrixNorm): Double {
        if (a.rows == 0 || a.cols == 0) return 0.0
        val code = when (kind) {
            MatrixNorm.ONE -> "1"
            MatrixNorm.INF -> "I"
            MatrixNorm.FROBENIUS -> "F"
            MatrixNorm.MAX -> "M"
        }
        return lapack.dlange(code, a.rows, a.cols, a.data, a.rows, DoubleArray(maxOf(1, a.rows)))
    }

    /** Отрицательный код возврата означает недопустимый аргумент вызова — ошибка программиста. */
    private fun checkArgs(routine: String, info: intW) {
        require(info.`val` >= 0) { "$routine: недопустимый аргумент номер ${-info.`val`}" }
    }
}
