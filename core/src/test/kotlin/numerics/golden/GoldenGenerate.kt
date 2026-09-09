package numerics.golden

import numerics.ConditionEstimate
import numerics.Conditioning
import numerics.GaussLegendre
import numerics.LinearAlgebra
import numerics.backend.Backends
import numerics.constCh
import numerics.golden.GoldenIo.dbl
import numerics.golden.GoldenIo.dblOrNull
import numerics.golden.GoldenIo.mat
import numerics.golden.GoldenIo.vec
import numerics.golden.GoldenInputs.conditionToJson
import numerics.golden.GoldenInputs.forwardErrorToJson
import numerics.golden.GoldenInputs.measuredToJson
import numerics.measured
import numerics.orderOrNull
import numerics.orders
import numerics.ratio
import numerics.reliableConstCh
import numerics.reliableOrders
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import java.io.File
import java.time.Instant

/**
 * Генератор golden-эталонов поведения версии 0.1.0. Не входит в обычный прогон
 * (`excludeTags("golden-generate")`); запускается задачей `./gradlew regenerateGolden`,
 * которая передаёт каталог назначения через `-Dgolden.dir`.
 */
@Tag("golden-generate")
class GoldenGenerate {

    private val dir: File
        get() = File(
            System.getProperty("golden.dir")
                ?: error("Не задано системное свойство golden.dir; используйте ./gradlew regenerateGolden"),
        )

    private fun meta(): Map<String, Any?> = linkedMapOf(
        "version" to (System.getProperty("golden.version") ?: "unknown"),
        "backend" to Backends.default().name,
        "date" to Instant.now().toString(),
    )

    private fun root(vararg sections: Pair<String, Any?>): Map<String, Any?> {
        val m = LinkedHashMap<String, Any?>()
        m["generatedWith"] = meta()
        for ((k, v) in sections) m[k] = v
        return m
    }

    @Test
    fun solve() {
        val cases = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.solveCases) {
            val a = case.matrix()
            val b = GoldenInputs.vec(case.n)
            val x = LinearAlgebra.solve(a, b)
            val omega = Conditioning.relativeBackwardError(a, b, x)
            val fe = LinearAlgebra.solveDiagnosed(a, b).forwardError
            cases[case.key] = linkedMapOf(
                "x" to vec(x),
                "omega" to dbl(omega),
                "forwardError" to forwardErrorToJson(fe),
            )
        }
        GoldenIo.writeGolden(dir, "solve.json", root("cases" to cases))
    }

    @Test
    fun matops() {
        val cases = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.matOpsCases) {
            val a = case.matrix()
            val b = GoldenInputs.vec(case.n)
            val aT = GoldenInputs.transpose(a)
            cases[case.key] = linkedMapOf(
                "matVec" to vec(LinearAlgebra.matVec(a, b)),
                "matTransVec" to vec(LinearAlgebra.matTransVec(a, b)),
                "matMat" to mat(LinearAlgebra.matMat(a, a)),
                "atWa" to mat(LinearAlgebra.atWa(a, GoldenInputs.posVec(case.n))),
                "addScaled" to mat(LinearAlgebra.addScaled(a, aT, 0.5)),
                "norm2" to dbl(LinearAlgebra.norm2(b)),
                "normInf" to dbl(LinearAlgebra.normInf(b)),
                "maxAsymmetry" to dbl(LinearAlgebra.maxAsymmetry(a)),
            )
        }
        GoldenIo.writeGolden(dir, "matops.json", root("cases" to cases))
    }

    @Test
    fun cholesky() {
        val cases = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.choleskyCases) {
            val l = LinearAlgebra.cholesky(case.matrix()) ?: error("cholesky(${case.key}) вернул null")
            cases[case.key] = linkedMapOf("L" to mat(l))
        }
        GoldenIo.writeGolden(dir, "cholesky.json", root("cases" to cases))
    }

    @Test
    fun conditioning() {
        val cases = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.conditioningCases) {
            val est = Conditioning.conditionInf(case.matrix())
            cases[case.key] = linkedMapOf(
                "condInf" to dbl(est.condInf),
                "inversionResidual" to dbl(est.inversionResidual),
                "isReliable" to est.isReliable,
            )
        }
        GoldenIo.writeGolden(dir, "conditioning.json", root("cases" to cases))
    }

    @Test
    fun eigen() {
        val eig = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.eigenCases) {
            eig[case.key] = vec(Conditioning.symmetricEigenvalues(case.matrix()).sortedArray())
        }
        val cond = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.conditionSymmetricCases) {
            cond[case.key] = dbl(Conditioning.conditionSymmetric(case.matrix()))
        }
        GoldenIo.writeGolden(dir, "eigen.json", root("symmetricEigenvalues" to eig, "conditionSymmetric" to cond))
    }

    @Test
    fun quadrature() {
        val reference = LinkedHashMap<String, Any?>()
        for (m in GoldenInputs.referenceOrders) {
            val (nodes, weights) = GaussLegendre.gaussLegendreReference(m)
            reference[m.toString()] = linkedMapOf("nodes" to vec(nodes), "weights" to vec(weights))
        }
        val integrate = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.integrateCases) {
            integrate[case.key] = dbl(GaussLegendre(case.m).integrate(case.bp, case.f))
        }
        GoldenIo.writeGolden(dir, "quadrature.json", root("reference" to reference, "integrate" to integrate))
    }

    @Test
    fun pure() {
        val fe = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.feCases) {
            val est = ConditionEstimate(case.cond, case.residual, 1e-8)
            fe[case.key] = linkedMapOf(
                "condition" to conditionToJson(est),
                "forwardError" to forwardErrorToJson(Conditioning.forwardError(est, case.omega)),
            )
        }

        val meas = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.measuredCases) {
            meas[case.key] = measuredToJson(measured(case.value, case.threshold))
        }

        val ord = LinkedHashMap<String, Any?>()
        for (list in GoldenInputs.orderLists) {
            ord[GoldenInputs.orderListKey(list)] = linkedMapOf(
                "input" to vec(list.toDoubleArray()),
                "orders" to vec(orders(list).toDoubleArray()),
                "reliableOrders" to reliableOrders(list).map { dblOrNull(it) },
            )
        }

        val cch = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.constChCases) {
            cch[case.key] = linkedMapOf(
                "constCh" to dbl(constCh(case.eh, case.h, case.p)),
                "reliableConstCh" to dblOrNull(reliableConstCh(measured(case.eh), case.h, case.p)),
            )
        }

        val rat = LinkedHashMap<String, Any?>()
        for (case in GoldenInputs.ratioCases) {
            rat[case.key] = linkedMapOf(
                "ratio" to dblOrNull(ratio(measured(case.a), measured(case.b))),
                "orderOrNull" to dblOrNull(orderOrNull(measured(case.a), measured(case.b))),
            )
        }

        GoldenIo.writeGolden(
            dir,
            "pure.json",
            root(
                "forwardError" to fe,
                "measured" to meas,
                "orders" to ord,
                "constCh" to cch,
                "ratio" to rat,
            ),
        )
    }
}
