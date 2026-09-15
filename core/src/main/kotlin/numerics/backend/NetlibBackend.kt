package numerics.backend

import dev.ludovic.netlib.blas.BLAS
import dev.ludovic.netlib.lapack.LAPACK
import dev.ludovic.netlib.lapack.NativeLAPACK
import numerics.DenseMatrix
import org.netlib.util.doubleW
import org.netlib.util.intW

/**
 * [LinAlgBackend] implementation over BLAS/LAPACK from the netlib library.
 *
 * The [blas]/[lapack] pair defines the computational kernel: either the system library via JNI
 * or the portable Java implementation; both share the same call interface, so a single
 * class serves both variants. [DenseMatrix] matrices are stored in column-major order,
 * and their arrays are passed to the kernel directly; where LAPACK overwrites its input with
 * the result, a copy is taken first — inputs are never modified.
 */
public class NetlibBackend internal constructor(
    private val blas: BLAS,
    private val lapack: LAPACK,
) : LinAlgBackend {

    override val isNative: Boolean = lapack is NativeLAPACK

    override val name: String =
        if (isNative) "netlib ${lapack.javaClass.simpleName} (native system BLAS/LAPACK)"
        else "netlib ${lapack.javaClass.simpleName} (Java)"

    override fun axpy(alpha: Double, x: DoubleArray, y: DoubleArray) {
        require(x.size == y.size) { "axpy: vector lengths must match, got ${x.size} and ${y.size}" }
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
            throw IllegalStateException("Matrix is singular: zero pivot at position ${info.`val`}")
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
        require(!lu.isSingular) { "LU factorization is singular: zero pivot at position ${lu.singularAt}" }
        val n = lu.lu.rows
        val bCopy = b.data.copyOf()
        val info = intW(0)
        lapack.dgetrs("N", n, b.cols, lu.lu.data, maxOf(1, n), lu.ipiv, bCopy, maxOf(1, n), info)
        checkArgs("dgetrs", info)
        return DenseMatrix.fromColumnMajor(n, b.cols, bCopy)
    }

    override fun inverse(a: DenseMatrix): DenseMatrix? {
        val lu = luFactor(a)
        if (lu.isSingular) return null
        return luSolve(lu, DenseMatrix.identity(a.rows))
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
        if (info.`val` > 0) throw IllegalStateException("Eigenvalue iteration failed to converge")
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

    /** A negative return code means an invalid call argument — a programming error. */
    private fun checkArgs(routine: String, info: intW) {
        require(info.`val` >= 0) { "$routine: invalid argument number ${-info.`val`}" }
    }
}
