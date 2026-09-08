package numerics.bench

// Микробенчмарк публичного API numerical-core: `LinearAlgebra` и `Conditioning`.
// Запуск: ./gradlew benchmark -Pbench.args="256 1024 2048".
//
// Используется только публичный API библиотеки — без прямых обращений к multik и к конкретным
// бэкендам, чтобы числа были сравнимы между версиями с разной внутренней реализацией.
// Все входы детерминированы (java.util.Random с seed = 1000 + n), поэтому таблицы разных
// запусков отличаются только временем.

import numerics.Conditioning
import numerics.LinearAlgebra
import numerics.backend.Backends
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.Random

/** Приёмник результатов — не даёт JIT выбросить вычисление как неиспользуемое. */
@Volatile
var sink: Any? = null

/** Случайная плотная матрица с диагональным доминированием: a[i][j] ∈ [-1, 1), a[i][i] += n. */
fun randomDD(n: Int): Array<DoubleArray> {
    val r = Random(1000L + n)
    return Array(n) { i ->
        val row = DoubleArray(n) { r.nextDouble() * 2 - 1 }
        row[i] += n.toDouble()
        row
    }
}

/** Случайный вектор с компонентами в [-1, 1); отдельный генератор, чтобы не зависеть от порядка вызовов. */
fun randomVec(n: Int): DoubleArray {
    val r = Random(2000L + n)
    return DoubleArray(n) { r.nextDouble() * 2 - 1 }
}

/** Точная симметризация: (b[i][j] + b[j][i]) / 2 записывается в обе позиции, асимметрия ровно 0. */
fun symmetrize(b: Array<DoubleArray>): Array<DoubleArray> {
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

/** Медиана времени (мс) по `reps` замерам после `warm` прогревочных запусков. */
fun median(warm: Int, reps: Int, block: () -> Any?): Double {
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

/** Время в мс с двумя знаками; `null` — измерение пропущено. */
private fun fmt(ms: Double?): String =
    if (ms == null) "—" else String.format(Locale.ROOT, "%.2f", ms)

fun main(args: Array<String>) {
    val sizes = (if (args.isEmpty()) listOf("256", "1024") else args.toList()).map { it.trim().toInt() }

    println("backend: ${Backends.default().name}")
    println("processors: ${Runtime.getRuntime().availableProcessors()}")
    println("jvm: ${System.getProperty("java.version")}")
    println("os.arch: ${System.getProperty("os.arch")}")
    println("date: ${ZonedDateTime.now().format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)}")
    println()
    println("Время в мс: медиана из 5 замеров после 3 прогревов (conditionInf и symmetricEigenvalues — медиана из 3).")
    println("conditionInf измеряется только при n ≤ 512, symmetricEigenvalues — при n ≤ 256; иначе «—» (слишком долго).")
    println()
    println("| n | solve | matMat | matVec | matTransVec | atWa | conditionInf | symmetricEigenvalues |")
    println("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for (n in sizes) {
        val a = randomDD(n)
        val b = randomVec(n)
        val solve = median(3, 5) { LinearAlgebra.solve(a, b) }
        val matMat = median(3, 5) { LinearAlgebra.matMat(a, a) }
        val matVec = median(3, 5) { LinearAlgebra.matVec(a, b) }
        val matTransVec = median(3, 5) { LinearAlgebra.matTransVec(a, b) }
        val atWa = median(3, 5) { LinearAlgebra.atWa(a, b) }
        // conditionInf в 0.1.0 стоит O(n) решений — при n = 1024 это десятки секунд, пропускаем.
        val cond = if (n <= 512) median(3, 3) { Conditioning.conditionInf(a) } else null
        // symmetricEigenvalues — циклический Якоби, при n = 1024 несколько секунд на вызов, пропускаем.
        val eig = if (n <= 256) {
            val s = symmetrize(a)
            median(3, 3) { Conditioning.symmetricEigenvalues(s) }
        } else {
            null
        }
        println(
            "| $n | ${fmt(solve)} | ${fmt(matMat)} | ${fmt(matVec)} | ${fmt(matTransVec)} | ${fmt(atWa)} | " +
                "${fmt(cond)} | ${fmt(eig)} |",
        )
        System.out.flush()
    }
    println()
    println("sink: ${sink?.javaClass?.simpleName}")
}
