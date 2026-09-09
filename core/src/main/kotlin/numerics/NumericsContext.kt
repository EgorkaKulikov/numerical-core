package numerics

import numerics.backend.Backends
import numerics.backend.LinAlgBackend

/**
 * Параметры вычислений: реализация линейной алгебры, разрешение параллельной сборки и степень
 * параллелизма. Неизменяемое значение; передаётся во все объекты одной задачи, чтобы результаты
 * были согласованы.
 *
 * Сравнение по значению (`data class`): два независимо созданных контекста с одинаковыми
 * параметрами описывают одну и ту же конфигурацию и считаются совместимыми.
 *
 * @param backend реализация линейной алгебры
 * @param parallel разрешена ли параллельная сборка ([ParallelAssembly])
 * @param parallelism число потоков параллельной сборки (не меньше 1)
 */
public data class NumericsContext(
    val backend: LinAlgBackend = Backends.default(),
    val parallel: Boolean = true,
    val parallelism: Int = Runtime.getRuntime().availableProcessors(),
) {
    init {
        require(parallelism >= 1) { "степень параллелизма должна быть не меньше 1, получено $parallelism" }
    }

    /** Разделяемый контекст по умолчанию и проверка совместимости контекстов. */
    /** Контекст по умолчанию и проверка совместимости контекстов. */
    public companion object {
        /**
         * Разделяемый экземпляр контекста по умолчанию: один на процесс, а не на каждый вызов
         * с параметром по умолчанию. `lazy`, потому что [Backends.default] может бросить
         * исключение, а исключение из инициализатора класса при повторном обращении
         * вырождается в `NoClassDefFoundError` без текста причины.
         */
        private val shared: Lazy<NumericsContext> = lazy { NumericsContext() }

        /** Контекст по умолчанию: стартовая реализация линейной алгебры, параллельная сборка включена. */
        public fun default(): NumericsContext = shared.value

        /**
         * Требует, чтобы зависимость использовала тот же контекст, что и владелец. Расхождение
         * контекстов означает, что части одной задачи будут посчитаны разными реализациями
         * линейной алгебры, что незаметно сдвигает младшие биты результата.
         *
         * @param owner имя класса-владельца для сообщения об ошибке
         * @param expected контекст владельца
         * @param dependency имя параметра-зависимости
         * @param actual контекст зависимости
         * @throws IllegalArgumentException если контексты различаются
         */
        public fun requireSame(
            owner: String,
            expected: NumericsContext,
            dependency: String,
            actual: NumericsContext,
        ) {
            require(expected == actual) {
                "Контекст вычислений у '$dependency' (${actual.describe()}) не совпадает с контекстом " +
                    "у '$owner' (${expected.describe()}). Передайте один и тот же NumericsContext " +
                    "во все связанные объекты."
            }
        }
    }

    /** Короткое описание для сообщений об ошибках. */
    public fun describe(): String = "backend=${backend.name}, parallel=$parallel, parallelism=$parallelism"
}
