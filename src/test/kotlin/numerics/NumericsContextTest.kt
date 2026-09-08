package numerics

import numerics.backend.Backends
import org.junit.jupiter.api.Tag
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Контракт [NumericsContext] как значения: разделяемый экземпляр по умолчанию, равенство
 * по значению и проверка совместимости [NumericsContext.requireSame].
 *
 * Проводка контекста через решатели интегральных уравнений (одинаковый результат на
 * разных бэкендах, громкий отказ при несовпадении контекстов семейства функционалов и
 * решателя) проверяется в репозитории `integral-equations`
 * (`numerics.NumericsContextWiringTest`): там живут сами решатели.
 */
@Tag("fast")
class NumericsContextTest {

    /** Контекст по умолчанию — РАЗДЕЛЯЕМЫЙ экземпляр, а не новый объект на каждый вызов. */
    @Test
    fun defaultContextIsSharedInstance() {
        assertSame(
            NumericsContext.default(), NumericsContext.default(),
            "NumericsContext.default() обязан отдавать один и тот же объект: иначе дефолтное " +
                "значение параметра аллоцировало бы объект на каждое построение решателя.",
        )
    }

    /** Равенство контекстов — ПО ЗНАЧЕНИЮ: два одинаково настроенных контекста совместимы. */
    @Test
    fun contextEqualityIsByValue() {
        assertEquals(NumericsContext(backend = Backends.java()), NumericsContext(backend = Backends.java()))
        assertTrue(NumericsContext(parallel = false) != NumericsContext(parallel = true))
    }

    /** [NumericsContext.requireSame] принимает равные по значению контексты и отвергает разные. */
    @Test
    fun requireSameAcceptsEqualAndRejectsDifferent() {
        val a = NumericsContext(backend = Backends.java(), parallel = false)
        val b = NumericsContext(backend = Backends.java(), parallel = false)
        NumericsContext.requireSame("Owner", a, "dep", b)

        val ex = assertFailsWith<IllegalArgumentException> {
            NumericsContext.requireSame("Owner", a, "dep", NumericsContext(backend = Backends.java(), parallel = true))
        }
        assertTrue(ex.message!!.contains("Owner"), "сообщение обязано называть владельца: ${ex.message}")
        assertTrue(ex.message!!.contains("'dep'"), "сообщение обязано называть зависимость: ${ex.message}")
    }
}
