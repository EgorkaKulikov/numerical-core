package numerics.backend

import dev.ludovic.netlib.blas.JavaBLAS
import dev.ludovic.netlib.blas.NativeBLAS
import dev.ludovic.netlib.lapack.JavaLAPACK
import dev.ludovic.netlib.lapack.NativeLAPACK
import java.util.logging.Level
import java.util.logging.Logger

/**
 * Выбор реализации линейной алгебры.
 *
 * Доступны две реализации [NetlibBackend]: [native] — системная BLAS/LAPACK через JNI
 * (Accelerate, OpenBLAS, MKL) и [java] — переносимая реализация на Java, которая
 * есть всегда. Реализация по умолчанию [default] выбирается один раз на процесс по
 * системному свойству `numerics.backend`:
 *  - `native` — только системная библиотека; если она не загрузилась, обращение
 *    к [default] бросает [IllegalStateException];
 *  - `java` — только реализация на Java;
 *  - `auto` (или свойство не задано) — системная библиотека при доступности, иначе
 *    Java с однократным предупреждением в журнал `numerics`.
 *
 * Выбор по умолчанию нельзя подменить после старта: кому нужна другая реализация,
 * передаёт её явно параметром [numerics.LinearAlgebra] или полем
 * [numerics.NumericsContext].
 */
public object Backends {

    private const val PROPERTY = "numerics.backend"

    /**
     * Ленивый стартовый выбор. `lazy` не запоминает исключение инициализатора, поэтому
     * при недоступной запрошенной реализации каждое обращение снова бросает исходное
     * исключение с исходным сообщением, а не `NoClassDefFoundError` без причины.
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
            throw IllegalStateException("Нативная реализация BLAS/LAPACK недоступна: ${e.message}", e)
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
     * Реализация по умолчанию, выбранная при первом обращении по свойству `numerics.backend`.
     * Обращение дёшево, поэтому безопасно как значение параметра по умолчанию.
     * @throws IllegalStateException если запрошена недоступная реализация.
     * @throws IllegalArgumentException если значение свойства не из числа допустимых.
     */
    public fun default(): LinAlgBackend = startup.value

    /**
     * Системная BLAS/LAPACK через JNI.
     * @throws IllegalStateException если нативная библиотека не загрузилась.
     */
    public fun native(): LinAlgBackend = nativeInstance.value

    /** Переносимая реализация на Java; доступна всегда. */
    public fun java(): LinAlgBackend = javaInstance.value

    /** Истина, если системная библиотека загрузилась; результат вычисляется один раз. */
    public fun isNativeAvailable(): Boolean = nativeAvailable.value

    /** Реально доступные реализации: системная (если загрузилась) и Java. */
    public fun available(): List<LinAlgBackend> =
        if (isNativeAvailable()) listOf(native(), java()) else listOf(java())

    /** Имя реализации по умолчанию и режим, заданный свойством. */
    public fun describe(): String =
        "backend=${default().name}, ${PROPERTY}=${System.getProperty(PROPERTY) ?: "auto"}"

    /**
     * Разбор значения свойства `numerics.backend` в реализацию; вынесен отдельно, чтобы
     * проверять логику выбора, не трогая стартовое значение [default].
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
                "$PROPERTY='$m': недопустимое значение; допустимые — native, java, auto"
            )
        }

    private val warned: Lazy<Unit> = lazy {
        System.getLogger("numerics").log(
            System.Logger.Level.WARNING,
            "Нативная BLAS/LAPACK не найдена, используется реализация на Java; производительность ниже",
        )
    }

    private fun warnOnce() = warned.value

    /** Гасит информационные сообщения netlib о выбранной реализации (пишутся в stderr). */
    private fun quietNetlibLogging() {
        for (name in listOf(
            "dev.ludovic.netlib",
            "dev.ludovic.netlib.blas.InstanceBuilder",
            "dev.ludovic.netlib.lapack.InstanceBuilder",
        )) Logger.getLogger(name).level = Level.WARNING
    }
}
