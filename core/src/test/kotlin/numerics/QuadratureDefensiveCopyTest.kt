package numerics

import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Encapsulation of the reference quadrature arrays: [GaussLegendre.refNodesWeights] must
 * return a copy so that a caller mutating the array cannot corrupt the source.
 *
 * If the internal arrays themselves were handed out, a single write into them would silently
 * change the quadrature for every user of the object — no exception, just different numbers.
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
        assertTrue(nodesBefore.contentEquals(nodesAgain), "quadrature nodes changed after mutating the copy")
        assertTrue(weightsBefore.contentEquals(weightsAgain), "quadrature weights changed after mutating the copy")
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
        assertEquals(before, after, 0.0, "integral changed after mutating the returned nodes/weights")
    }
}
