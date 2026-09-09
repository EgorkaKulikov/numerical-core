package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend
import numerics.backend.MatrixNorm
import numerics.golden.GoldenInputs
import org.junit.jupiter.api.Assertions.assertTimeoutPreemptively
import org.junit.jupiter.api.Assumptions.assumeTrue
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.DynamicTest.dynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import org.junit.jupiter.api.function.ThrowingSupplier
import java.time.Duration
import kotlin.math.abs
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Граничные случаи публичного API: реестр реализаций, матрицы 1×1, прямоугольные и пустые
 * матрицы, большие размеры, вырожденность разной структуры, численно экстремальные масштабы,
 * крайние параметры квадратуры и параллельной сборки, границы [measured].
 *
 * Каждый случай — отдельный DynamicTest; операции линейной алгебры гоняются на обеих
 * реализациях (нативная — через `assumeTrue(isNativeAvailable())`).
 */
@Tag("fast")
class EdgeCasesTest {

    private fun m(vararg rows: DoubleArray): DenseMatrix = DenseMatrix.fromRows(arrayOf(*rows))
    private fun v(vararg x: Double): DoubleArray = doubleArrayOf(*x)

    private fun assertVec(expected: DoubleArray, actual: DoubleArray, tol: Double = 1e-12, what: String = "") {
        assertEquals(expected.size, actual.size, "$what: длина")
        for (i in expected.indices) assertTrue(abs(expected[i] - actual[i]) <= tol, "$what[$i]: ${expected[i]} vs ${actual[i]}")
    }

    /** Один DynamicTest на реализацию; для нативной — пропуск, если она не загрузилась. */
    private fun perBackend(name: String, body: (LinAlgBackend) -> Unit): List<DynamicTest> = listOf(
        dynamicTest("$name [java]") { body(Backends.java()) },
        dynamicTest("$name [native]") {
            assumeTrue(Backends.isNativeAvailable(), "нативная BLAS/LAPACK недоступна")
            body(Backends.native())
        },
    )

    private fun <T> cases(items: List<Pair<String, T>>, body: (LinAlgBackend, T) -> Unit): List<DynamicTest> =
        items.flatMap { (name, item) -> perBackend(name) { b -> body(b, item) } }

    // --- Реестр реализаций -----------------------------------------------------------

    @TestFactory
    fun registry(): List<DynamicTest> = listOf(
        dynamicTest("available() непустой и содержит java()") {
            val all = Backends.available()
            assertTrue(all.isNotEmpty())
            assertTrue(all.any { it === Backends.java() }, "java() должен быть в available()")
        },
        dynamicTest("available() содержит native() ровно тогда, когда isNativeAvailable()") {
            val hasNative = Backends.available().any { it.isNative }
            assertEquals(Backends.isNativeAvailable(), hasNative)
            if (hasNative) assertTrue(Backends.available().any { it === Backends.native() })
        },
        dynamicTest("имена реализаций в available() различны") {
            val names = Backends.available().map { it.name }
            assertEquals(names.size, names.toSet().size, "дубли имён: $names")
        },
        dynamicTest("describe() содержит имя реализации по умолчанию") {
            assertTrue(Backends.describe().contains(Backends.default().name), Backends.describe())
        },
        dynamicTest("resolve(java) — не нативная") { assertTrue(!Backends.resolve("java").isNative) },
        dynamicTest("resolve(native) — нативная") {
            assumeTrue(Backends.isNativeAvailable())
            assertTrue(Backends.resolve("native").isNative)
        },
        dynamicTest("resolve(auto) — нативная iff доступна") {
            assertEquals(Backends.isNativeAvailable(), Backends.resolve("auto").isNative)
        },
        dynamicTest("resolve(multik) — IllegalArgumentException с перечислением допустимых") {
            val e = assertFailsWith<IllegalArgumentException> { Backends.resolve("multik") }
            val msg = e.message!!
            assertTrue(msg.contains("native") && msg.contains("java") && msg.contains("auto"), msg)
        },
    )

    // --- Матрицы 1×1 -------------------------------------------------------------------

    @TestFactory
    fun oneByOne(): List<DynamicTest> = perBackend("1×1") { b ->
        assertVec(v(2.0), LinearAlgebra.solve(arrayOf(v(2.0)), v(4.0), b), what = "solve")
        assertVec(v(0.25), Conditioning.inverse(m(v(4.0)), b)!!.data, what = "inverse")
        assertVec(v(3.0), LinearAlgebra.cholesky(m(v(9.0)), b)!!.data, what = "cholesky")
        assertVec(v(7.0), Conditioning.symmetricEigenvalues(m(v(7.0)), b), what = "eig")
        assertEquals(1.0, Conditioning.conditionInf(m(v(1.0)), backend = b).condInf, 1e-15, "condInf")
        assertVec(v(6.0), LinearAlgebra.matMat(m(v(2.0)), m(v(3.0)), b).data, what = "matMat")
    }

    // --- Прямоугольные -------------------------------------------------------------------

    @TestFactory
    fun rectangular(): List<DynamicTest> = perBackend("прямоугольные") { b ->
        val a35 = DenseMatrix.build(3, 5) { i, j -> (i + 2 * j).toDouble() }
        assertEquals(3, LinearAlgebra.matVec(a35, DoubleArray(5) { 1.0 }, b).size, "matVec 3×5·5")
        assertEquals(5, LinearAlgebra.matTransVec(a35, DoubleArray(3) { 1.0 }, b).size, "matTransVec 3×5ᵀ·3")
        val a23 = DenseMatrix.build(2, 3) { i, j -> (i + j).toDouble() }
        val a34 = DenseMatrix.build(3, 4) { i, j -> (i - j).toDouble() }
        val p = LinearAlgebra.matMat(a23, a34, b)
        assertEquals(2, p.rows); assertEquals(4, p.cols)
        assertFailsWith<IllegalArgumentException>("matMat 2×3·2×3") { LinearAlgebra.matMat(a23, a23, b) }
        assertFailsWith<IllegalArgumentException>("solve 3×4") { LinearAlgebra.solve(a34, v(1.0, 2.0, 3.0), b) }
        assertFailsWith<IllegalArgumentException>("cholesky 3×4") { LinearAlgebra.cholesky(a34, b) }
        for (k in MatrixNorm.values()) {
            val n = b.norm(a35, k)
            assertTrue(n.isFinite() && n >= 0.0, "norm $k = $n")
        }
    }

    // --- Пустые ---------------------------------------------------------------------------

    @TestFactory
    fun empty(): List<DynamicTest> = perBackend("пустые") { b ->
        assertFailsWith<IllegalArgumentException>("matVec 0×0") {
            LinearAlgebra.matVec(DenseMatrix.zeros(0, 0), DoubleArray(0), b)
        }
        assertEquals(0, DenseMatrix.zeros(0, 5).toRows().size)
        assertEquals(0, DenseMatrix.fromRows(emptyArray()).rows)
    }

    // --- Большие размеры ---------------------------------------------------------------------

    @TestFactory
    fun large(): List<DynamicTest> {
        val native = Backends.isNativeAvailable()
        val b = if (native) Backends.native() else Backends.java()
        val nSolve = if (native) 1000 else 300
        val nEig = if (native) 500 else 200
        val tag = if (native) "native" else "java"
        return listOf(
            dynamicTest("solve на диагонально доминирующей $nSolve×$nSolve [$tag]") {
                val a = DenseMatrix.fromRows(GoldenInputs.dd(nSolve))
                val rhs = GoldenInputs.vec(nSolve)
                val x = LinearAlgebra.solve(a, rhs, b)
                val r = LinearAlgebra.matVec(a, x, b)
                var res = 0.0
                for (i in rhs.indices) res = maxOf(res, abs(r[i] - rhs[i]))
                val scale = b.norm(a, MatrixNorm.INF) * LinearAlgebra.normInf(x) + LinearAlgebra.normInf(rhs)
                assertTrue(res <= 1e-10 * scale, "невязка $res при масштабе $scale")
            },
            dynamicTest("symmetricEigenvalues на симметричной $nEig×$nEig: сумма = след [$tag]") {
                val s = DenseMatrix.fromRows(GoldenInputs.sym(nEig))
                val eig = Conditioning.symmetricEigenvalues(s, b)
                var trace = 0.0
                for (i in 0 until nEig) trace += s[i, i]
                assertEquals(nEig, eig.size)
                assertTrue(abs(eig.sum() - trace) <= 1e-8 * b.norm(s, MatrixNorm.INF), "Σλ=${eig.sum()} vs tr=$trace")
            },
            dynamicTest("conditionEstimate на диагонально доминирующей $nSolve×$nSolve за разумное время [$tag]") {
                val a = DenseMatrix.fromRows(GoldenInputs.dd(nSolve))
                val est = assertTimeoutPreemptively(Duration.ofSeconds(10), ThrowingSupplier { Conditioning.conditionEstimate(a, backend = b) })
                assertTrue(est.condInf.isFinite() && est.condInf >= 1.0, "cond=${est.condInf}")
            },
        )
    }

    // --- Вырожденные разной структуры -----------------------------------------------------------

    @TestFactory
    fun singularStructures(): List<DynamicTest> = perBackend("вырожденные") { b ->
        val zero = DenseMatrix.zeros(3, 3)
        assertFailsWith<IllegalStateException>("solve нулевой") { LinearAlgebra.solve(zero, v(1.0, 2.0, 3.0), b) }
        assertNull(Conditioning.inverse(zero, b), "inverse нулевой")
        assertEquals(Double.POSITIVE_INFINITY, Conditioning.conditionInf(zero, backend = b).condInf, "conditionInf нулевой")
        assertEquals(Double.POSITIVE_INFINITY, Conditioning.conditionEstimate(zero, backend = b).condInf, "conditionEstimate нулевой")
        val rankDeficient = arrayOf(v(1.0, 2.0, 3.0), v(2.0, 4.0, 6.0), v(1.0, 1.0, 1.0))
        assertFailsWith<IllegalStateException>("ранг-дефицит") { LinearAlgebra.solve(rankDeficient, v(1.0, 2.0, 3.0), b) }
        // Нулевой диагональный элемент при невырожденной матрице — перестановка, а не вырожденность.
        assertVec(v(2.0, 1.0), LinearAlgebra.solve(arrayOf(v(0.0, 1.0), v(1.0, 0.0)), v(1.0, 2.0), b), what = "перестановка")
    }

    // --- Численно экстремальные --------------------------------------------------------------------

    @TestFactory
    fun extremeScales(): List<DynamicTest> = cases(
        listOf("1e300·I" to 1e300, "1e-300·I" to 1e-300),
    ) { b, scale ->
        val a = arrayOf(v(scale, 0.0, 0.0), v(0.0, scale, 0.0), v(0.0, 0.0, scale))
        val rhs = v(1.0, 2.0, 3.0)
        val x = LinearAlgebra.solve(a, DoubleArray(3) { rhs[it] * scale }, b)
        assertVec(rhs, x, tol = 1e-15, what = "scale=$scale")
    } + perBackend("Double.MIN_VALUE на диагонали") { b ->
        val a = arrayOf(v(Double.MIN_VALUE, 0.0), v(0.0, Double.MIN_VALUE))
        try {
            val x = LinearAlgebra.solve(a, v(1.0, 1.0), b)
            assertTrue(x.all { it.isFinite() }, "без исключения решение обязано быть числовым: ${x.toList()}")
        } catch (_: IllegalStateException) {
            // Допустимый исход: реализация честно отказалась.
        }
    }

    // --- Квадратура ----------------------------------------------------------------------------

    @TestFactory
    fun quadrature(): List<DynamicTest> = listOf(
        dynamicTest("m=64: Σw=2, узлы в (−1, 1)") {
            val (x, w) = GaussLegendre.gaussLegendreReference(64)
            assertEquals(2.0, w.sum(), 1e-14)
            assertTrue(x.all { it > -1.0 && it < 1.0 })
            val (x2, w2) = GaussLegendre(64).refNodesWeights()
            assertVec(x, x2, 0.0, "refNodes"); assertVec(w, w2, 0.0, "refWeights")
        },
        dynamicTest("m=200: Σw=2 в 1e-13, узлы в (−1, 1)") {
            val (x, w) = GaussLegendre.gaussLegendreReference(200)
            assertEquals(2.0, w.sum(), 1e-13)
            assertTrue(x.all { it > -1.0 && it < 1.0 })
            assertEquals(200, x.toSet().size, "узлы различны")
        },
        dynamicTest("отрезок длины 1e-12 — конечный результат") {
            val r = GaussLegendre(8).integrate(v(0.0, 1e-12)) { t -> t * t + 1.0 }
            assertTrue(r.isFinite() && abs(r - 1e-12) <= 1e-24, "r=$r")
        },
        dynamicTest("отрезок [−1e6, 1e6] от единицы — 2e6") {
            assertEquals(2e6, GaussLegendre(8).integrate(v(-1e6, 1e6)) { 1.0 }, 2e6 * 1e-9)
        },
    )

    // --- Параллельная сборка ---------------------------------------------------------------------

    @TestFactory
    fun parallelAssembly(): List<DynamicTest> = listOf(
        dynamicTest("assembleDense 1×100000 совпадает с последовательным") {
            val f = { i: Int, j: Int -> i * 3.0 + j * 0.5 }
            val par = ParallelAssembly.assembleDense(1, 100_000, cellFn = f)
            val seq = ParallelAssembly.assembleDense(1, 100_000, parallel = false, cellFn = f)
            assertTrue(par.data.contentEquals(seq.data))
        },
        dynamicTest("assembleDense 100000×1 совпадает с последовательным") {
            val f = { i: Int, j: Int -> i * 3.0 + j * 0.5 }
            val par = ParallelAssembly.assembleDense(100_000, 1, cellFn = f)
            val seq = ParallelAssembly.assembleDense(100_000, 1, parallel = false, cellFn = f)
            assertTrue(par.data.contentEquals(seq.data))
        },
        dynamicTest("вложенный assembleDense на общем пуле не зависает") {
            val nested = { i: Int, j: Int ->
                ParallelAssembly.assembleDense(3, 3) { a, c -> (a + c).toDouble() }.data.sum() + i + j
            }
            val par = assertTimeoutPreemptively(Duration.ofSeconds(30), ThrowingSupplier { ParallelAssembly.assembleDense(4, 4, cellFn = nested) })
            val seq = ParallelAssembly.assembleDense(4, 4, parallel = false, cellFn = nested)
            assertTrue(par.data.contentEquals(seq.data))
        },
    )

    // --- measured ----------------------------------------------------------------------------------

    @TestFactory
    fun measuredBoundaries(): List<DynamicTest> = listOf(
        dynamicTest("measured(-0.0) — AtNoiseLevel, как 0.0") { assertIs<Measured.AtNoiseLevel>(measured(-0.0)) },
        dynamicTest("measured(Double.MIN_VALUE) — AtNoiseLevel") { assertIs<Measured.AtNoiseLevel>(measured(Double.MIN_VALUE)) },
        dynamicTest("measured(1e-13, 1e-13) — граница ВКЛЮЧЕНА: |value| >= threshold считается надёжным") {
            assertIs<Measured.Reliable>(measured(1e-13, 1e-13))
            assertIs<Measured.AtNoiseLevel>(measured(Math.nextDown(1e-13), 1e-13))
        },
        dynamicTest("measured(NaN) — AtNoiseLevel, значение сохранено") {
            val r = measured(Double.NaN)
            assertNotNull(r)
            assertIs<Measured.AtNoiseLevel>(r)
            assertTrue(r.value.isNaN())
        },
    )
}
