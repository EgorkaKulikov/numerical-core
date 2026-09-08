package numerics

import numerics.backend.Backends
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.DynamicTest.dynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.math.exp
import kotlin.math.sin
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/** Валидация входов и договорённости о знаке/порядке: квадратура, порядки сходимости, сборка, контекст. */
@Tag("fast")
class InputValidationTest {

    @TestFactory
    fun quadrature(): List<DynamicTest> = listOf(
        dynamicTest("GaussLegendre(0) отклоняется") {
            assertFailsWith<IllegalArgumentException> { GaussLegendre(0) }
        },
        dynamicTest("gaussLegendreReference(0) и (-1) отклоняются") {
            assertFailsWith<IllegalArgumentException> { GaussLegendre.gaussLegendreReference(0) }
            assertFailsWith<IllegalArgumentException> { GaussLegendre.gaussLegendreReference(-1) }
        },
        dynamicTest("integrate: разбиение из одной точки отклоняется") {
            assertFailsWith<IllegalArgumentException> { GaussLegendre(4).integrate(doubleArrayOf(0.0)) { it } }
        },
        dynamicTest("integrate: невозрастающее разбиение отклоняется с указанием причины") {
            val ex = assertFailsWith<IllegalArgumentException> {
                GaussLegendre(4).integrate(doubleArrayOf(0.0, 1.0, 0.5)) { it }
            }
            assertTrue(ex.message!!.contains("возрастать"), "сообщение: ${ex.message}")
        },
        dynamicTest("integrateInterval: перестановка концов меняет знак побитово") {
            val q = GaussLegendre(8)
            val forward = q.integrateInterval(0.0, 1.0, ::exp)
            val backward = q.integrateInterval(1.0, 0.0, ::exp)
            assertEquals((-forward).toBits(), backward.toBits())
        },
        dynamicTest("integrateInterval: вырожденный отрезок даёт 0") {
            assertEquals(0.0, GaussLegendre(8).integrateInterval(0.3, 0.3, ::exp))
        },
        dynamicTest("integrate по разбиению равен сумме по подынтервалам") {
            val q = GaussLegendre(8)
            val whole = q.integrate(doubleArrayOf(0.0, 0.5, 1.0)) { sin(it) }
            val parts = q.integrateInterval(0.0, 0.5) { sin(it) } + q.integrateInterval(0.5, 1.0) { sin(it) }
            assertTrue(kotlin.math.abs(whole - parts) <= 1e-15, "whole=$whole parts=$parts")
        },
    )

    @TestFactory
    fun convergenceRates(): List<DynamicTest> = listOf(
        dynamicTest("orders([Inf, 1.0]) даёт NaN в обеих позициях") {
            val p = orders(listOf(Double.POSITIVE_INFINITY, 1.0))
            assertTrue(p.all { it.isNaN() }, "получено $p")
        },
        dynamicTest("orders([1.0, Inf]) даёт NaN в обеих позициях") {
            val p = orders(listOf(1.0, Double.POSITIVE_INFINITY))
            assertTrue(p.all { it.isNaN() }, "получено $p")
        },
        dynamicTest("orders([1.0, 0.5, 0.25]) даёт [1, 1, NaN]") {
            val p = orders(listOf(1.0, 0.5, 0.25))
            assertEquals(3, p.size)
            assertEquals(1.0, p[0], 1e-15)
            assertEquals(1.0, p[1], 1e-15)
            assertTrue(p[2].isNaN())
        },
        dynamicTest("constCh с неположительным шагом отклоняется") {
            assertFailsWith<IllegalArgumentException> { constCh(1e-3, 0.0, 2.0) }
            assertFailsWith<IllegalArgumentException> { constCh(1e-3, -0.1, 2.0) }
        },
        dynamicTest("reliableConstCh с нулевым шагом отклоняется") {
            assertFailsWith<IllegalArgumentException> { reliableConstCh(measured(1e-3), 0.0, 2.0) }
        },
    )

    @TestFactory
    fun parallelAssembly(): List<DynamicTest> = listOf(
        dynamicTest("assembleDense с отрицательным размером отклоняется") {
            assertFailsWith<IllegalArgumentException> { ParallelAssembly.assembleDense(-1, 2) { _, _ -> 0.0 } }
        },
        dynamicTest("assembleRows с отрицательным размером отклоняется") {
            assertFailsWith<IllegalArgumentException> { ParallelAssembly.assembleRows(-1, 2) { DoubleArray(2) } }
        },
        dynamicTest("assembleDense(0, 0) даёт пустую матрицу") {
            val m = ParallelAssembly.assembleDense(0, 0) { _, _ -> 1.0 }
            assertEquals(0, m.rows)
            assertEquals(0, m.cols)
        },
        dynamicTest("результат сборки не зависит от степени параллелизма") {
            val cellFn = { i: Int, j: Int -> sin(i * 37.0 + j) }
            val reference = ParallelAssembly.assembleDense(37, 11, parallel = true, cellFn)
            val one = ParallelAssembly.assembleDense(37, 11, NumericsContext(parallelism = 1), cellFn)
            val three = ParallelAssembly.assembleDense(37, 11, NumericsContext(parallelism = 3), cellFn)
            val seq = ParallelAssembly.assembleDense(37, 11, NumericsContext(parallel = false), cellFn)
            assertTrue(reference.data.contentEquals(one.data), "parallelism=1")
            assertTrue(reference.data.contentEquals(three.data), "parallelism=3")
            assertTrue(reference.data.contentEquals(seq.data), "parallel=false")
        },
    )

    @TestFactory
    fun numericsContext(): List<DynamicTest> = listOf(
        dynamicTest("parallelism = 0 отклоняется") {
            assertFailsWith<IllegalArgumentException> { NumericsContext(parallelism = 0) }
        },
        dynamicTest("requireSame называет оба объекта и не упоминает решатель") {
            val a = NumericsContext(backend = Backends.java(), parallel = false)
            val b = NumericsContext(backend = Backends.java(), parallel = true)
            val ex = assertFailsWith<IllegalArgumentException> {
                NumericsContext.requireSame("Owner", a, "dep", b)
            }
            val msg = ex.message!!
            assertTrue(msg.contains("'Owner'"), msg)
            assertTrue(msg.contains("'dep'"), msg)
            assertFalse(msg.lowercase().contains("решател"), msg)
        },
        dynamicTest("describe содержит степень параллелизма") {
            assertTrue(NumericsContext(parallelism = 2).describe().contains("parallelism=2"))
        },
        dynamicTest("перегрузка solve с контекстом совпадает побитово с перегрузкой с backend") {
            val a = DenseMatrix.build(5, 5) { i, j -> if (i == j) 4.0 else 1.0 / (1 + i + j) }
            val b = DoubleArray(5) { it + 1.0 }
            val viaContext = LinearAlgebra.solve(a, b, NumericsContext(backend = Backends.java()))
            val viaBackend = LinearAlgebra.solve(a, b, Backends.java())
            assertContentEquals(viaBackend, viaContext)
        },
    )
}
