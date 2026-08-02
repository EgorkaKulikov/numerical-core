package numerics.backend

/**
 * Реестр и селектор подключаемых бэкендов линейной алгебры.
 *
 * Здесь регистрируются реализации [LinAlgBackend] в порядке приоритета.
 * Именно сюда будущий GPU/HPC-бэкенд добавляется одной строкой в список
 * [candidates] — код решателей и фасад [numerics.LinearAlgebra] остаются
 * нетронутыми.
 *
 * Выбор бэкенда ПО УМОЛЧАНИЮ (того, что отдаёт [default] и что попадает в
 * [numerics.NumericsContext] по умолчанию):
 *  - если задано системное свойство `numerics.backend` ("multik" | "reference"),
 *    выбирается ИМЕННО он; если запрошенный бэкенд недоступен на этой машине —
 *    бросается [IllegalStateException], а НЕ выполняется молчаливая подмена
 *    (см. [selectInitial]);
 *  - если свойство не задано — выбирается ПЕРВЫЙ из [candidates], у которого
 *    [LinAlgBackend.isAvailable] вернул true.
 *
 * Таким образом, при отсутствии нативного OpenBLAS и БЕЗ явного запроса происходит
 * автоматический откат на [ReferenceBackend] (чистый JVM) — это и есть история
 * переносимости для HPC: предпочесть ускоренный бэкенд, но гарантировать
 * работоспособность.
 *
 * Реестр НЕИЗМЕНЯЕМ: подменить выбранный бэкенд глобально нельзя — кому нужен
 * другой, передаёт его явно через [numerics.NumericsContext] или параметром
 * [numerics.LinearAlgebra].
 */
object Backends {

    /** Кандидаты в порядке приоритета: ускоренный CPU, затем чистый JVM fallback. */
    private val candidates: List<LinAlgBackend> = listOf(MultikCpuBackend, ReferenceBackend)

    /**
     * ЛЕНИВЫЙ стартовый выбор бэкенда.
     *
     * Ленивость здесь — не оптимизация, а требование к диагностике. Если выполнять
     * [selectInitial] в инициализаторе объекта, то при `-Dnumerics.backend=nosuchbackend`
     * внятный [IllegalStateException] вылетает ровно один раз, завёрнутый в
     * `ExceptionInInitializerError`, а КАЖДОЕ последующее обращение к [Backends] даёт
     * `NoClassDefFoundError: Could not initialize class numerics.backend.Backends`
     * уже БЕЗ текста причины — пользователь видит непонятную ошибку загрузки класса
     * вместо объяснения, что он опечатался в имени бэкенда.
     *
     * `lazy` (режим SYNCHRONIZED) не запоминает исключение инициализатора: при неудаче
     * значение остаётся невычисленным, и следующее обращение снова выполняет выбор,
     * снова бросая ИСХОДНОЕ исключение с исходным сообщением.
     */
    private val startup: Lazy<LinAlgBackend> = lazy { selectInitial() }

    /**
     * Бэкенд ПО УМОЛЧАНИЮ: выбранный при старте процесса по `-Dnumerics.backend`
     * либо авто-выбором.
     *
     * Это НЕ глобальное мутируемое состояние: значение вычисляется один раз и
     * подменить его нельзя. Кому нужен другой бэкенд — передаёт его ЯВНО:
     * параметром [numerics.LinearAlgebra] или полем [numerics.NumericsContext].
     * Прежние `use()`/`reset()` удалены: они делали результат вычисления зависимым
     * от того, что успел выставить сосед по JVM.
     *
     * Функция, а не свойство: обращение может БРОСИТЬ (неизвестный или недоступный
     * запрошенный бэкенд), а бросающий геттер читается как дешёвое чтение поля.
     * Сам вызов дёшев (чтение уже вычисленного [startup]), поэтому его безопасно
     * использовать значением параметра по умолчанию в горячем пути.
     *
     * @throws IllegalStateException при КАЖДОМ обращении, если стартовый выбор
     *   невозможен (запрошен неизвестный или недоступный бэкенд).
     */
    fun default(): LinAlgBackend = startup.value

    /** Список зарегистрированных бэкендов (в порядке приоритета). */
    fun available(): List<LinAlgBackend> = candidates

    /**
     * Начальный выбор бэкенда.
     *
     * ЯВНЫЙ запрос через `-Dnumerics.backend=...` исполняется буквально: если
     * запрошенный бэкенд недоступен, поднимается ошибка. Молчаливый откат в этом
     * случае — источник трудноуловимых расхождений: на плохо обусловленных задачах
     * (например, регуляризованное уравнение первого рода) замена multik на reference
     * меняет численный результат на проценты, а пользователь, задавший свойство
     * осознанно, считает, что работает на запрошенном бэкенде.
     *
     * При ОТСУТСТВИИ свойства авто-выбор с откатом сохранён: там он уместен и
     * является заявленной гарантией переносимости.
     *
     * @throws IllegalStateException если запрошен недоступный или неизвестный бэкенд.
     */
    private fun selectInitial(): LinAlgBackend =
        select(System.getProperty("numerics.backend")?.trim()?.lowercase())

    /**
     * Чистая (тестируемая) логика выбора — та же, что применяется при старте.
     *
     * @param requested значение свойства `numerics.backend` в нижнем регистре либо null.
     * @param isAvailable предикат доступности; параметризован, чтобы тест мог
     *        воспроизвести машину БЕЗ нативной библиотеки, не подменяя саму библиотеку.
     */
    internal fun select(
        requested: String?,
        isAvailable: (LinAlgBackend) -> Boolean = { it.isAvailable() },
    ): LinAlgBackend {
        if (!requested.isNullOrEmpty()) {
            val backend = when (requested) {
                "multik" -> MultikCpuBackend
                "reference" -> ReferenceBackend
                else -> error(
                    "numerics.backend='$requested': неизвестный бэкенд; " +
                        "допустимые значения — multik, reference"
                )
            }
            check(isAvailable(backend)) {
                "numerics.backend='$requested': бэкенд '${backend.name}' недоступен на этой машине " +
                    "(нативная библиотека не загрузилась). Молчаливая подмена запрещена: она меняет " +
                    "численные результаты. Уберите -Dnumerics.backend, чтобы разрешить авто-выбор " +
                    "с откатом на '${ReferenceBackend.name}'."
            }
            return backend
        }
        return candidates.firstOrNull { isAvailable(it) } ?: ReferenceBackend
    }
}
