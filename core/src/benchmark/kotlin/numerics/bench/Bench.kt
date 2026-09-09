package numerics.bench

import numerics.Conditioning
import numerics.DenseMatrix
import numerics.LinearAlgebra
import numerics.backend.Backends
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.Random

// Бенчмарк основных операций на DenseMatrix. Входы — те же генераторы, что и в golden-эталонах
// (java.util.Random(1000 + n) для матрицы, Random(2000 + n) для вектора); матрица строго
// диагонально доминирующая. Преобразование в DenseMatrix выполняется один раз вне измеряемого блока.

@Volatile
private var sink: Any? = null

private fun randomDD(n: Int): Array<DoubleArray> {
    val r = Random(1000L + n)
    return Array(n) { i ->
        val row = DoubleArray(n) { r.nextDouble() * 2 - 1 }
        row[i] += n.toDouble()
        row
    }
}

private fun randomVec(n: Int): DoubleArray {
    val r = Random(2000L + n)
    return DoubleArray(n) { r.nextDouble() * 2 - 1 }
}

private fun symmetrize(b: Array<DoubleArray>): Array<DoubleArray> {
    val n = b.size
    val s = Array(n) { DoubleArray(n) }
    for (i in 0 until n) {
        for (j in i until n) {
            val v = (b[i][j] + b[j][i]) / 2
            s[i][j] = v
            s[j][i] = v
        }
    }
    return s
}

private fun median(warm: Int, reps: Int, block: () -> Any?): Double {
    repeat(warm) { sink = block() }
    val t = DoubleArray(reps)
    for (k in 0 until reps) {
        val t0 = System.nanoTime()
        sink = block()
        t[k] = (System.nanoTime() - t0) / 1e6
    }
    t.sort()
    return t[reps / 2]
}

private fun fmt(ms: Double?): String =
    if (ms == null) "—" else String.format(Locale.ROOT, "%.2f", ms)

/** Точка входа микробенчмарка; аргументы — размеры матриц, по умолчанию 256 и 1024. */
public fun main(args: Array<String>) {
    val sizes = (if (args.isEmpty()) listOf("256", "1024") else args.toList()).map { it.trim().toInt() }
    println("backend: ${Backends.default().name}")
    println("processors: ${Runtime.getRuntime().availableProcessors()}")
    println("jvm: ${System.getProperty("java.version")}")
    println("os.arch: ${System.getProperty("os.arch")}")
    println("date: ${ZonedDateTime.now().format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)}")
    println()
    println("Время в мс: медиана из 5 замеров после 3 прогревов (conditionInf и symmetricEigenvalues — медиана из 3).")
    println("conditionInf и symmetricEigenvalues измеряются только при n ≤ 1024; иначе «—».")
    println()
    println("| n | solve | matMat | matVec | matTransVec | atWa | conditionInf | conditionEstimate | symmetricEigenvalues |")
    println("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (n in sizes) {
        val rows = randomDD(n)
        val dense = DenseMatrix.fromRows(rows)
        val b = randomVec(n)
        val w = DoubleArray(n) { 1.0 + (it % 7) * 0.1 }
        val solve = median(3, 5) { LinearAlgebra.solve(dense, b) }
        val matMat = median(3, 5) { LinearAlgebra.matMat(dense, dense) }
        val matVec = median(3, 5) { LinearAlgebra.matVec(dense, b) }
        val matTransVec = median(3, 5) { LinearAlgebra.matTransVec(dense, b) }
        val atWa = median(3, 5) { LinearAlgebra.atWa(dense, w) }
        val condInf = if (n <= 1024) median(3, 3) { Conditioning.conditionInf(dense) } else null
        val condEst = median(3, 5) { Conditioning.conditionEstimate(dense) }
        val eig = if (n <= 1024) {
            val symDense = DenseMatrix.fromRows(symmetrize(rows))
            median(3, 3) { Conditioning.symmetricEigenvalues(symDense) }
        } else {
            null
        }
        println(
            "| $n | ${fmt(solve)} | ${fmt(matMat)} | ${fmt(matVec)} | ${fmt(matTransVec)} | ${fmt(atWa)} | " +
                "${fmt(condInf)} | ${fmt(condEst)} | ${fmt(eig)} |",
        )
        System.out.flush()
    }
    println()
    println("sink: ${sink?.javaClass?.simpleName}")
}
