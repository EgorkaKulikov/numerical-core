package numerics.backend

/**
 * Реестр и селектор подключаемых бэкендов линейной алгебры.
 *
 * Здесь регистрируются реализации [LinAlgBackend] в порядке приоритета.
 * Именно сюда будущий GPU/HPC-бэкенд добавляется одной строкой в список
 * [candidates] — код решателей и фасад [numerics.LinearAlgebra] остаются
 * нетронутыми.
 *
 * Выбор активного бэкенда:
 *  - если задано системное свойство `numerics.backend` ("multik" | "reference"),
 *    выбирается соответствующий бэкенд, если он доступен;
 *  - иначе выбирается ПЕРВЫЙ из [candidates], у которого [LinAlgBackend.isAvailable]
 *    вернул true.
 *
 * Таким образом, при отсутствии нативного OpenBLAS происходит автоматический
 * откат на [ReferenceBackend] (чистый JVM) — это и есть история переносимости
 * для HPC: предпочесть ускоренный бэкенд, но гарантировать работоспособность.
 */
object Backends {

    /** Кандидаты в порядке приоритета: ускоренный CPU, затем чистый JVM fallback. */
    private val candidates: List<LinAlgBackend> = listOf(MultikCpuBackend, ReferenceBackend)

    /** Активный бэкенд, которому делегирует фасад [numerics.LinearAlgebra]. */
    @Volatile
    var active: LinAlgBackend = selectInitial()
        private set

    /** Список зарегистрированных бэкендов (в порядке приоритета). */
    fun available(): List<LinAlgBackend> = candidates

    /** Явно установить активный бэкенд (для тестов/бенчмарков). */
    fun use(backend: LinAlgBackend) {
        active = backend
    }

    /**
     * Начальный выбор: при заданном `-Dnumerics.backend=...` уважаем его,
     * иначе авто-выбор первого доступного кандидата.
     */
    private fun selectInitial(): LinAlgBackend {
        val requested = System.getProperty("numerics.backend")?.trim()?.lowercase()
        when (requested) {
            "multik" -> if (MultikCpuBackend.isAvailable()) return MultikCpuBackend
            "reference" -> return ReferenceBackend
        }
        return candidates.firstOrNull { it.isAvailable() } ?: ReferenceBackend
    }
}
