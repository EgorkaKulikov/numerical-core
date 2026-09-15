package numerics.openblas

import org.bytedeco.javacpp.Loader
import org.bytedeco.openblas.global.openblas_full
import java.io.File

/**
 * Hooks the bundled BLAS/LAPACK implementation (OpenBLAS) into the `numerical-core` library.
 *
 * This module is for machines without a system BLAS/LAPACK. `numerical-core` looks up the
 * native implementation through the system properties `dev.ludovic.netlib.blas.nativeLibPath` and
 * `dev.ludovic.netlib.lapack.nativeLibPath`, reading them once at first initialization.
 * Therefore [install] must be called before the first access to `numerics.backend.Backends` and before
 * any library call; otherwise the pure JVM implementation is selected.
 *
 * Example:
 * ```
 * fun main() {
 *     OpenBlas.install()
 *     println(Backends.describe())
 * }
 * ```
 *
 * Stack requirement: with more than one thread, OpenBLAS runs the LU factorization
 * multithreaded and consumes the calling thread's stack. The default stack of the JVM main thread
 * may be insufficient, in which case the JVM crashes inside native code. Large systems should
 * be solved either with `-Xss4m` (or larger) or on a thread with an explicit
 * stack size: `Thread(null, r, "lapack", 8L shl 20)`.
 */
public object OpenBlas {
    private const val BLAS_PROP = "dev.ludovic.netlib.blas.nativeLibPath"
    private const val LAPACK_PROP = "dev.ludovic.netlib.lapack.nativeLibPath"

    private val logger = System.getLogger("numerics.openblas")

    /**
     * Default number of OpenBLAS threads: the smaller of the available processor count and four.
     *
     * The upper bound is needed for two reasons: on processors with heterogeneous cores
     * (performance and efficiency) more threads slow factorizations down rather than speed them up;
     * moreover, multithreaded LU factorization consumes the calling thread's stack, and the more
     * threads, the larger the stack it requires.
     */
    public val defaultThreads: Int = minOf(Runtime.getRuntime().availableProcessors(), 4)

    /**
     * Path to the extracted OpenBLAS library (with LAPACK), or `null` if it could not be
     * extracted for the current platform.
     *
     * The first call extracts the library from the jar into the user cache and loads it,
     * so it may take noticeable time. Among the files next to the loaded library, the one with
     * the shortest name is chosen that starts with `libopenblas` and is neither
     * the LAPACK-less variant (`nolapack`) nor the JNI wrapper.
     */
    public fun libraryPath(): String? = try {
        val jni = File(Loader.load(openblas_full::class.java))
        val dir = jni.parentFile
        dir?.listFiles()
            ?.filter { f ->
                val n = f.name
                n.startsWith("libopenblas") &&
                    !n.contains("nolapack") &&
                    !n.contains("jni") &&
                    (n.endsWith(".dylib") || n.endsWith(".so") || n.endsWith(".dll") || n.contains(".so."))
            }
            ?.sortedBy { it.name.length }
            ?.firstOrNull()
            ?.absolutePath
    } catch (e: UnsatisfiedLinkError) {
        null
    } catch (e: Exception) {
        null
    }

    /**
     * Returns `true` if the system properties with the native BLAS and LAPACK path are already set
     * (by this module or manually).
     */
    public fun isInstalled(): Boolean =
        System.getProperty(LAPACK_PROP) != null && System.getProperty(BLAS_PROP) != null

    /**
     * Extracts OpenBLAS, records its path in the system properties and sets the thread count.
     *
     * Call before the first access to `numerics.backend.Backends` and any library call:
     * the path is read once at initialization. A repeated call changes nothing and returns
     * `true`; properties already set manually are left untouched as well.
     *
     * With [threads] greater than one the calling thread needs a stack of at least 4 MB (`-Xss4m` or
     * a thread with an explicit stack size); otherwise the JVM may crash when solving
     * large systems.
     *
     * @param threads number of OpenBLAS threads, at least 1; defaults to [defaultThreads].
     * @return `true` if the native implementation is hooked in; `false` if the library could not
     * be extracted for the current platform (the library keeps running on the JVM implementation).
     */
    public fun install(threads: Int = defaultThreads): Boolean {
        require(threads >= 1) { "Thread count must be at least 1, got $threads" }
        if (isInstalled()) return true
        val path = libraryPath() ?: return false
        System.setProperty(BLAS_PROP, path)
        System.setProperty(LAPACK_PROP, path)
        openblas_full.openblas_set_num_threads(threads)
        if (threads > 1) {
            logger.log(
                System.Logger.Level.INFO,
                "OpenBLAS: $threads threads; a thread solving large systems needs a stack of at least 4 MB"
            )
        }
        return true
    }
}
