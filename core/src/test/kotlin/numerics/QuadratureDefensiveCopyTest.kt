package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Инкапсуляция эталонных массивов квадратуры: [GaussLegendre.refNodesWeights] обязан
 * отдавать копию, чтобы изменение массива у вызывающего не портило источник.
 *
 * Если бы наружу уходили сами внутренние массивы, одна запись в них незаметно меняла бы
 * квадратуру сразу для всех пользователей объекта — без исключения, просто получались бы
 * другие числа.
 */
@Tag("fast")
class QuadratureDefensiveCopyTest {

    @Test fun refNodesWeightsReturnsCopy() {
        val quad = GaussLegendre(8)
        val (nodes, weights) = quad.refNodesWeights()
        val nodesBefore = nodes.copyOf()
        val weightsBefore = weights.copyOf()

        nodes[0] = 12345.0
        weights[0] = -777.0

        val (nodesAgain, weightsAgain) = quad.refNodesWeights()
        assertTrue(nodesBefore.contentEquals(nodesAgain), "узлы квадратуры изменились после мутации копии")
        assertTrue(weightsBefore.contentEquals(weightsAgain), "веса квадратуры изменились после мутации копии")
    }

    @Test fun mutatingRefNodesDoesNotAffectIntegral() {
        val quad = GaussLegendre(8)
        val bp = doubleArrayOf(0.0, 0.5, 1.0)
        val exact = 1.0 / 3.0
        val before = quad.integrate(bp) { t -> t * t }

        val (nodes, weights) = quad.refNodesWeights()
        for (i in nodes.indices) { nodes[i] = 0.0; weights[i] = 0.0 }

        val after = quad.integrate(bp) { t -> t * t }
        assertEquals(exact, before, 1e-14)
        assertEquals(before, after, 0.0, "интеграл изменился после мутации возвращённых узлов/весов")
    }
}
