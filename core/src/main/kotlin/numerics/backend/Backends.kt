package numerics.backend

import dev.ludovic.netlib.blas.JavaBLAS
import dev.ludovic.netlib.blas.NativeBLAS
import dev.ludovic.netlib.lapack.JavaLAPACK
import dev.ludovic.netlib.lapack.NativeLAPACK
import java.util.logging.Level
import java.util.logging.Logger

/**
 * Selection of the linear algebra backend.
 *
 * Two [NetlibBackend] flavours are available: [native] — the system BLAS/LAPACK via JNI
 * (Accelerate, OpenBLAS, MKL) and [java] — a portable pure-Java implementation that is
 * always present. The default backend [default] is chosen once per process from the
 * system property `numerics.backend`:
 *  - `native` — the system library only; if it failed to load, calling [default]
 *    throws [IllegalStateException];
 *  - `java` — the Java implementation only;
 *  - `auto` (or property not set) — the system library when available, otherwise
 *    Java with a one-time warning to the `numerics` logger.
 *
 * The default cannot be replaced after startup: code that needs a different backend
 * passes it explicitly as a [numerics.LinearAlgebra] parameter or via a
 * [numerics.NumericsContext] field.
 */
public object Backends {

    private const val PROPERTY = "numerics.backend"

    /**
     * Lazy startup selection. `lazy` does not cache the initializer's exception, so when
     * the requested backend is unavailable every access rethrows the original exception
     * with its original message rather than a `NoClassDefFoundError` without a cause.
     */
    private val startup: Lazy<LinAlgBackend> = lazy { resolve(System.getProperty(PROPERTY)) }

    private val javaInstance: Lazy<LinAlgBackend> = lazy {
        quietNetlibLogging()
        NetlibBackend(JavaBLAS.getInstance(), JavaLAPACK.getInstance())
    }

    private val nativeInstance: Lazy<LinAlgBackend> = lazy {
        quietNetlibLogging()
        try {
            NetlibBackend(NativeBLAS.getInstance(), NativeLAPACK.getInstance())
        } catch (e: RuntimeException) {
            throw IllegalStateException("Native BLAS/LAPACK implementation is unavailable: ${e.message}", e)
        }
    }

    private val nativeAvailable: Lazy<Boolean> = lazy {
        try {
            nativeInstance.value
            true
        } catch (_: IllegalStateException) {
            false
        }
    }

    /**
     * The default backend, chosen on first access from the `numerics.backend` property.
     * Access is cheap, so it is safe to use as a default parameter value.
     * @throws IllegalStateException if the requested backend is unavailable.
     * @throws IllegalArgumentException if the property value is not one of the allowed ones.
     */
    public fun default(): LinAlgBackend = startup.value

    /**
     * The system BLAS/LAPACK via JNI.
     * @throws IllegalStateException if the native library failed to load.
     */
    public fun native(): LinAlgBackend = nativeInstance.value

    /** Portable Java implementation; always available. */
    public fun java(): LinAlgBackend = javaInstance.value

    /** True if the system library has loaded; the result is computed once. */
    public fun isNativeAvailable(): Boolean = nativeAvailable.value

    /** Backends actually available: the system one (if loaded) and Java. */
    public fun available(): List<LinAlgBackend> =
        if (isNativeAvailable()) listOf(native(), java()) else listOf(java())

    /** Name of the default backend and the mode set by the property. */
    public fun describe(): String =
        "backend=${default().name}, ${PROPERTY}=${System.getProperty(PROPERTY) ?: "auto"}"

    /**
     * Parses the `numerics.backend` property value into a backend; kept separate so the
     * selection logic can be tested without touching the startup value of [default].
     */
    internal fun resolve(mode: String?): LinAlgBackend =
        when (val m = mode?.trim()?.lowercase()) {
            null, "", "auto" -> if (isNativeAvailable()) native() else {
                warnOnce()
                java()
            }
            "native" -> native()
            "java" -> java()
            else -> throw IllegalArgumentException(
                "$PROPERTY='$m': invalid value; allowed values are native, java, auto"
            )
        }

    private val warned: Lazy<Unit> = lazy {
        System.getLogger("numerics").log(
            System.Logger.Level.WARNING,
            "Native BLAS/LAPACK not found, falling back to the Java implementation; performance will be lower",
        )
    }

    private fun warnOnce() = warned.value

    /** Silences netlib's informational messages about the chosen implementation (written to stderr). */
    private fun quietNetlibLogging() {
        for (name in listOf(
            "dev.ludovic.netlib",
            "dev.ludovic.netlib.blas.InstanceBuilder",
            "dev.ludovic.netlib.lapack.InstanceBuilder",
        )) Logger.getLogger(name).level = Level.WARNING
    }
}
