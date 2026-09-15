package numerics

import numerics.backend.Backends
import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Contract of [NumericsContext] as a value: a shared default instance, equality
 * by value and the compatibility check [NumericsContext.requireSame].
 */
@Tag("fast")
class NumericsContextTest {

    /** The default context is a shared instance, not a new object on every call. */
    @Test
    fun defaultContextIsSharedInstance() {
        assertSame(
            NumericsContext.default(), NumericsContext.default(),
            "NumericsContext.default() must return the same object: otherwise the default " +
                "parameter value would allocate an object on every call using the default.",
        )
    }

    /** Context equality is by value: two identically configured contexts are compatible. */
    @Test
    fun contextEqualityIsByValue() {
        assertEquals(NumericsContext(backend = Backends.java()), NumericsContext(backend = Backends.java()))
        assertTrue(NumericsContext(parallel = false) != NumericsContext(parallel = true))
    }

    /** [NumericsContext.requireSame] accepts value-equal contexts and rejects different ones. */
    @Test
    fun requireSameAcceptsEqualAndRejectsDifferent() {
        val a = NumericsContext(backend = Backends.java(), parallel = false)
        val b = NumericsContext(backend = Backends.java(), parallel = false)
        NumericsContext.requireSame("Owner", a, "dep", b)

        val ex = assertFailsWith<IllegalArgumentException> {
            NumericsContext.requireSame("Owner", a, "dep", NumericsContext(backend = Backends.java(), parallel = true))
        }
        assertTrue(ex.message!!.contains("Owner"), "message must name the owner: ${ex.message}")
        assertTrue(ex.message!!.contains("'dep'"), "message must name the dependency: ${ex.message}")
    }
}
