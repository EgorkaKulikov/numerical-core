package numerics.golden

import numerics.ConditionEstimate
import numerics.Conditioning
import numerics.ForwardError
import numerics.Measured
import numerics.constCh
import numerics.golden.GoldenIo.assertBits
import numerics.golden.GoldenIo.assertBitsOrNull
import numerics.golden.GoldenIo.entry
import numerics.golden.GoldenIo.obj
import numerics.golden.GoldenIo.section
import numerics.golden.GoldenIo.toDbl
import numerics.golden.GoldenIo.toDblOrNull
import numerics.golden.GoldenIo.toVec
import numerics.measured
import numerics.orderOrNull
import numerics.orders
import numerics.ratio
import numerics.reliableConstCh
import numerics.reliableOrders
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.TestFactory
import kotlin.test.assertEquals

/** Эталон `pure.json`: чистые функции без LAPACK — всё сравнивается побитово. */
@Tag("fast")
class GoldenPureTest {

    @TestFactory
    fun cases(): List<DynamicTest> {
        val root = GoldenIo.readGolden("pure.json")
        val fe = section(root, "forwardError")
        val meas = section(root, "measured")
        val ord = section(root, "orders")
        val cch = section(root, "constCh")
        val rat = section(root, "ratio")

        val feTests = GoldenInputs.feCases.map { case ->
            DynamicTest.dynamicTest("forwardError ${case.key}") {
                val exp = obj(entry(fe, case.key))
                val est = ConditionEstimate(case.cond, case.residual, 1e-8)
                assertConditionBits(obj(exp["condition"]), est, "${case.key}.condition")
                assertForwardErrorBits(obj(exp["forwardError"]), Conditioning.forwardError(est, case.omega), case.key)
            }
        }

        val measTests = GoldenInputs.measuredCases.map { case ->
            DynamicTest.dynamicTest("measured ${case.key}") {
                assertMeasuredBits(obj(entry(meas, case.key)), measured(case.value, case.threshold), case.key)
            }
        }

        val ordTests = GoldenInputs.orderLists.map { list ->
            val key = GoldenInputs.orderListKey(list)
            DynamicTest.dynamicTest("orders $key") {
                val exp = obj(entry(ord, key))
                val expInput = toVec(exp["input"])
                assertEquals(list.size, expInput.size, "$key.input.size")
                for (i in list.indices) assertBits(expInput[i], list[i], "$key.input[$i]")

                val expOrders = toVec(exp["orders"])
                val gotOrders = orders(list)
                assertEquals(expOrders.size, gotOrders.size, "$key.orders.size")
                for (i in gotOrders.indices) assertBits(expOrders[i], gotOrders[i], "$key.orders[$i]")

                val expRel = (exp["reliableOrders"] as List<*>).map { toDblOrNull(it) }
                val gotRel = reliableOrders(list)
                assertEquals(expRel.size, gotRel.size, "$key.reliableOrders.size")
                for (i in gotRel.indices) assertBitsOrNull(expRel[i], gotRel[i], "$key.reliableOrders[$i]")
            }
        }

        val cchTests = GoldenInputs.constChCases.map { case ->
            DynamicTest.dynamicTest("constCh ${case.key}") {
                val exp = obj(entry(cch, case.key))
                assertBits(toDbl(exp["constCh"]), constCh(case.eh, case.h, case.p), "${case.key}.constCh")
                assertBitsOrNull(
                    toDblOrNull(exp["reliableConstCh"]),
                    reliableConstCh(measured(case.eh), case.h, case.p),
                    "${case.key}.reliableConstCh",
                )
            }
        }

        val ratTests = GoldenInputs.ratioCases.map { case ->
            DynamicTest.dynamicTest("ratio ${case.key}") {
                val exp = obj(entry(rat, case.key))
                val a = measured(case.a)
                val b = measured(case.b)
                assertBitsOrNull(toDblOrNull(exp["ratio"]), ratio(a, b), "${case.key}.ratio")
                assertBitsOrNull(toDblOrNull(exp["orderOrNull"]), orderOrNull(a, b), "${case.key}.orderOrNull")
            }
        }

        return feTests + measTests + ordTests + cchTests + ratTests
    }

    private fun assertConditionBits(exp: Map<String, Any?>, got: ConditionEstimate, label: String) {
        assertBits(toDbl(exp["condInf"]), got.condInf, "$label.condInf")
        assertBits(toDbl(exp["inversionResidual"]), got.inversionResidual, "$label.inversionResidual")
        assertBits(toDbl(exp["tolerance"]), got.tolerance, "$label.tolerance")
        assertEquals(exp["isReliable"] as Boolean, got.isReliable, "$label.isReliable")
    }

    private fun assertForwardErrorBits(exp: Map<String, Any?>, got: ForwardError, label: String) {
        val gotType = when (got) {
            is ForwardError.Bounded -> "Bounded"
            is ForwardError.NoFiniteBound -> "NoFiniteBound"
            is ForwardError.Unreliable -> "Unreliable"
        }
        assertEquals(exp["type"] as String, gotType, "$label.type")
        assertBits(toDbl(exp["backwardError"]), got.backwardError, "$label.backwardError")
        when (got) {
            is ForwardError.Bounded -> {
                assertBits(toDbl(exp["cond"]), got.cond, "$label.cond")
                assertBits(toDbl(exp["relativeBound"]), got.relativeBound, "$label.relativeBound")
            }
            is ForwardError.NoFiniteBound -> Unit
            is ForwardError.Unreliable -> assertConditionBits(obj(exp["condition"]), got.condition, "$label.condition")
        }
    }

    private fun assertMeasuredBits(exp: Map<String, Any?>, got: Measured, label: String) {
        val gotType = when (got) {
            is Measured.Reliable -> "Reliable"
            is Measured.AtNoiseLevel -> "AtNoiseLevel"
        }
        assertEquals(exp["type"] as String, gotType, "$label.type")
        assertBits(toDbl(exp["value"]), got.value, "$label.value")
        if (got is Measured.AtNoiseLevel) {
            assertBits(toDbl(exp["threshold"]), got.threshold, "$label.threshold")
        }
    }
}
