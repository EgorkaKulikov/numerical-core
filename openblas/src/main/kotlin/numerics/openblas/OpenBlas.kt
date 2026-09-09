package numerics.openblas

import org.bytedeco.javacpp.Loader
import org.bytedeco.openblas.global.openblas_full
import java.io.File

/**
 * Подключение упакованной реализации BLAS/LAPACK (OpenBLAS) к библиотеке `numerical-core`.
 *
 * Модуль нужен на машинах, где системной BLAS/LAPACK нет. Библиотека `numerical-core` ищет
 * нативную реализацию через системные свойства `dev.ludovic.netlib.blas.nativeLibPath` и
 * `dev.ludovic.netlib.lapack.nativeLibPath`, читая их один раз при первой инициализации.
 * Поэтому [install] надо вызвать до первого обращения к `numerics.backend.Backends` и до любого
 * вызова библиотеки, иначе будет выбрана чистая JVM-реализация.
 *
 * Пример:
 * ```
 * fun main() {
 *     OpenBlas.install()
 *     println(Backends.describe())
 * }
 * ```
 *
 * Требование к стеку: при числе потоков больше одного OpenBLAS выполняет LU-разложение
 * многопоточно и расходует стек вызывающего потока. Стандартного стека главного потока JVM
 * может не хватить, и тогда JVM аварийно останавливается внутри нативного кода. Решать большие
 * системы следует либо с параметром `-Xss4m` (или больше), либо в потоке с явно заданным
 * размером стека: `Thread(null, r, "lapack", 8L shl 20)`.
 */
public object OpenBlas {
    private const val BLAS_PROP = "dev.ludovic.netlib.blas.nativeLibPath"
    private const val LAPACK_PROP = "dev.ludovic.netlib.lapack.nativeLibPath"

    private val logger = System.getLogger("numerics.openblas")

    /**
     * Число потоков OpenBLAS по умолчанию: меньшее из числа доступных процессоров и четырёх.
     *
     * Ограничение сверху нужно по двум причинам: на процессорах с разнородными ядрами
     * (быстрые и энергоэффективные) большее число потоков замедляет разложения, а не ускоряет;
     * кроме того, многопоточное LU-разложение расходует стек вызывающего потока, и чем больше
     * потоков, тем выше требования к его размеру.
     */
    public val defaultThreads: Int = minOf(Runtime.getRuntime().availableProcessors(), 4)

    /**
     * Путь к распакованной библиотеке OpenBLAS (с LAPACK) или `null`, если для текущей
     * платформы её распаковать не удалось.
     *
     * Первый вызов распаковывает библиотеку из jar-файла в кэш пользователя и загружает её,
     * поэтому может занять заметное время. Среди файлов рядом с загруженной библиотекой
     * выбирается вариант с самым коротким именем, начинающийся с `libopenblas` и не являющийся
     * ни вариантом без LAPACK (`nolapack`), ни обёрткой для JNI.
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
     * Возвращает `true`, если системные свойства с путём к нативной BLAS и LAPACK уже заданы
     * (этим модулем или вручную).
     */
    public fun isInstalled(): Boolean =
        System.getProperty(LAPACK_PROP) != null && System.getProperty(BLAS_PROP) != null

    /**
     * Распаковывает OpenBLAS, прописывает путь к ней в системные свойства и задаёт число потоков.
     *
     * Вызывать до первого обращения к `numerics.backend.Backends` и любому вызову библиотеки:
     * путь читается один раз при инициализации. Повторный вызов ничего не меняет и возвращает
     * `true`; если свойства уже заданы вручную, они тоже не трогаются.
     *
     * При [threads] больше одного вызывающему потоку нужен стек не менее 4 МБ (`-Xss4m` или
     * поток с явным размером стека), иначе при решении больших систем возможен аварийный
     * останов JVM.
     *
     * @param threads число потоков OpenBLAS, не меньше 1; по умолчанию [defaultThreads].
     * @return `true`, если нативная реализация подключена; `false`, если распаковать библиотеку
     * для текущей платформы не удалось (библиотека продолжит работать на JVM-реализации).
     */
    public fun install(threads: Int = defaultThreads): Boolean {
        require(threads >= 1) { "число потоков должно быть не меньше 1, получено $threads" }
        if (isInstalled()) return true
        val path = libraryPath() ?: return false
        System.setProperty(BLAS_PROP, path)
        System.setProperty(LAPACK_PROP, path)
        openblas_full.openblas_set_num_threads(threads)
        if (threads > 1) {
            logger.log(
                System.Logger.Level.INFO,
                "OpenBLAS: потоков $threads; потоку, решающему большие системы, нужен стек не менее 4 МБ"
            )
        }
        return true
    }
}
