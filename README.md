# numerical-core

Набор численных примитивов для плотных вычислений на Kotlin/JVM. Библиотека использует
BLAS/LAPACK целевой системы (Intel MKL, AMD AOCL, OpenBLAS, Apple Accelerate) и
предназначена для программ, которым нужны решение систем линейных уравнений с оценкой
достоверности результата, число обусловленности, спектр симметричных матриц, разложение
Холецкого, составная квадратура Гаусса–Лежандра и параллельная сборка матриц — с одинаковым
поведением на рабочей станции и на вычислительном узле кластера.

Версия 1.0.0. Kotlin 2.0, JDK 21+. Лицензия Apache License 2.0.

## Состав

| Класс или пакет | Назначение |
|---|---|
| `DenseMatrix` | Плотная матрица в столбцовом порядке; передаётся в BLAS/LAPACK без копирования. Конструкторы `zeros`, `identity`, `diagonal`, `fromRows`, `fromColumnMajor`, `build`. |
| `LinearAlgebra` | Единая точка входа: `solve`, `solveDiagnosed`, `matVec`, `matTransVec`, `matMat`, `atWa`, `addScaled`, `cholesky`, нормы `norm2`, `normInf`, `maxAsymmetry`. |
| `Conditioning`, `ConditionEstimate`, `ForwardError`, `ConditionSource` | Число обусловленности через обращение или оценку LAPACK, спектр симметричных матриц, обратная и прямая ошибка решения с признаком достоверности. |
| `GaussLegendre` | Составная квадратура Гаусса–Лежандра по произвольному разбиению отрезка. |
| `ParallelAssembly` | Параллельная сборка матриц по строкам или по элементам: `assembleRows`, `assembleMatrix`, `assembleDense`. |
| `NumericsContext` | Реализация линейной алгебры, разрешение параллелизма и его степень — одним значением, передаваемым по цепочке вызовов. |
| `Measured`, `measured`, `reliableOrders`, `reliableConstCh`, `orders`, `constCh` | Отделение измеренной величины от шума округления; наблюдаемый порядок сходимости и константа `C·h^p`. |
| пакет `numerics.backend` | `LinAlgBackend`, `Backends`, `MatrixNorm`, `LuFactorization`, `NetlibBackend` — доступ к BLAS/LAPACK и выбор реализации. |
| модуль `numerical-core-openblas` | `OpenBlas.install()` — упакованная OpenBLAS для систем без собственной библиотеки. |

## Три способа получить BLAS/LAPACK

Библиотека обращается к BLAS/LAPACK через netlib и выбирает реализацию при первом
обращении.

1. **Системная библиотека** — режим по умолчанию. На macOS используется Apple Accelerate,
   на Linux — `liblapack.so.3` и `libblas.so.3`, которые предоставляют пакеты
   `libopenblas-dev`, Intel MKL или AMD AOCL через механизм альтернатив. Путь к библиотекам
   можно задать явно системными свойствами `dev.ludovic.netlib.blas.nativeLibPath` и
   `dev.ludovic.netlib.lapack.nativeLibPath`.
2. **Упакованная OpenBLAS** — модуль `numerical-core-openblas`. Вызов `OpenBlas.install()`
   до первого обращения к библиотеке распаковывает OpenBLAS для текущей платформы и
   указывает на неё netlib. Подробности и ограничения — в `openblas/README.md`.
3. **Переносимая реализация на Java** — используется автоматически, если нативная
   библиотека не найдена. Работает везде, где есть JVM, но медленнее в 10–20 раз; библиотека
   сообщает о переходе на неё предупреждением в журнал.

Режим выбирается системным свойством `-Dnumerics.backend=native|java|auto` (по умолчанию
`auto`); `Backends.describe()` возвращает строку с именем выбранной реализации и признаком
того, нативная ли она. Поведение всех методов — проверки, исключения, форматы результата —
одинаково для любой реализации.

## Подключение

Требуется JDK 21 или новее. Артефакты публикуются в локальный репозиторий Maven:

    ./gradlew publishToMavenLocal

```kotlin
repositories { mavenLocal(); mavenCentral() }

dependencies {
    implementation("io.github.egorkakulikov:numerical-core:1.0.0")
    // при необходимости упакованной OpenBLAS:
    implementation("io.github.egorkakulikov:numerical-core-openblas:1.0.0")
}
```

## Пример

```kotlin
import numerics.*
import numerics.backend.Backends
import kotlin.math.PI
import kotlin.math.sin

fun main() {
    val a = DenseMatrix.fromRows(arrayOf(
        doubleArrayOf( 2.0, -1.0,  0.0),
        doubleArrayOf(-1.0,  2.0, -1.0),
        doubleArrayOf( 0.0, -1.0,  2.0),
    ))
    val b = doubleArrayOf(1.0, 0.0, 1.0)

    // Решение с проверкой невязки: вырожденная или нечисловая система отвергается исключением.
    val x: DoubleArray = LinearAlgebra.solve(a, b)

    // Решение с оценкой прямой ошибки.
    val d = LinearAlgebra.solveDiagnosed(a, b, source = ConditionSource.ESTIMATE)
    when (val e = d.forwardError) {
        is ForwardError.Bounded       -> println("‖x − x*‖/‖x*‖ ≤ ${e.relativeBound}, cond ≈ ${e.cond}")
        is ForwardError.NoFiniteBound -> println("система численно вырождена, ω = ${e.backwardError}")
        is ForwardError.Unreliable    -> println("обусловленность не определена достоверно")
    }

    // Число обусловленности: оценка LAPACK за O(n²) после разложения.
    val cond: ConditionEstimate = Conditioning.conditionEstimate(a)
    println("cond ≈ ${cond.valueOrNull()}")

    // Составная квадратура Гаусса–Лежандра: ∫₀^π sin t dt = 2.
    val integral = GaussLegendre(8).integrate(doubleArrayOf(0.0, PI / 2, PI)) { t -> sin(t) }

    // Параллельная сборка матрицы по элементам.
    val h = ParallelAssembly.assembleDense(200, 200) { i, j -> 1.0 / (i + j + 1) }

    println(Backends.describe())
}
```

## Точность и достоверность

`LinearAlgebra.solve` вычисляет относительную обратную ошибку
`ω = ‖Ax − b‖∞ / max(‖A‖∞‖x‖∞, ‖b‖∞)` и отвергает решение, если `ω > 10⁻¹⁰` или во входе
есть нечисловые значения; для обратно устойчивого LU-разложения этого достаточно, чтобы
обнаружить вырожденность и сбой реализации. `solveDiagnosed` дополнительно возвращает
`ForwardError`: `Bounded` — относительная ошибка не превышает `cond · ω`; `NoFiniteBound` —
`cond · ω ≥ 1`, граница не несёт информации; `Unreliable` — сама оценка обусловленности
недостоверна. Недостоверная величина никогда не возвращается как число. Тип `Measured`
отделяет измеренные значения от шума округления (порог `10⁻¹³`), а `reliableOrders`
вычисляет порядок сходимости только по достоверным измерениям. Подробности и обоснование
порогов — в `docs/ТОЧНОСТЬ.md`.

## Производительность

Все операции `O(n³)` выполняет BLAS/LAPACK; данные передаются без копирования, потоки
внутри операций линейной алгебры создаёт нативная библиотека и управляет ими через свои
переменные окружения (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`).
На Apple M1 Pro с Accelerate: `solve` для `n = 1024` — 11.5 мс (в версии 0.1.0 — 37.7 мс),
`conditionInf` для `n = 256` — 1.5 мс (в версии 0.1.0 — 294 мс). Таблица измерений, режим
переносимой реализации и рекомендации для модуля OpenBLAS — в `docs/ПРОИЗВОДИТЕЛЬНОСТЬ.md`.

## Проверка

    ./gradlew build                                              # тесты обоих модулей, покрытие Kover
    ./gradlew :numerical-core:test -Dnumerics.backend=java       # тесты на переносимой реализации
    ./gradlew :numerical-core:test -Dnumerics.backend=native     # тесты на системной библиотеке
    ./gradlew :numerical-core:benchmark -Pbench.args="256 1024"  # измерения производительности
    ./gradlew :numerical-core:dokkaHtml                          # документация API

| Вид тестов | Что проверяют |
|---|---|
| Эталоны поведения (golden, `core/src/test/resources/golden`) | Результаты на фиксированных входах не меняются между версиями и реализациями. |
| Свойства на случайных матрицах (jqwik) | Совпадение с двумя независимыми оракулами — Hipparchus и собственной реализацией LU в тестах. |
| Граничные случаи | Вырожденные, нечисловые, несимметричные и пустые входы; матрицы Гильберта. |
| Потокобезопасность | Параллельные вызовы через одну реализацию линейной алгебры. |
| Обе реализации | Одни и те же тесты выполняются с `-Dnumerics.backend=java` и `native`. |

## Документация

- `docs/ТОЧНОСТЬ.md` — обратная и прямая ошибка, число обусловленности, пороги и их обоснование.
- `docs/ПРОИЗВОДИТЕЛЬНОСТЬ.md` — путь данных, многопоточность, таблица измерений.
- `docs/ИСТОЧНИКИ.md` — происхождение каждого алгоритма и список литературы.
- `docs/РЕФЕРАТ.md` — реферат для заявки на государственную регистрацию программы для ЭВМ.
- `openblas/README.md` — модуль с упакованной OpenBLAS.
- `CONTRIBUTING.md` — правила изменения кода и документации.
- KDoc публичного API — `./gradlew :numerical-core:dokkaHtml`, результат в `core/build/dokka/html`.

## Лицензия

Apache License 2.0. Правообладатель — Егор Куликов. Текст лицензии — в файле `LICENSE`,
уведомления о сторонних компонентах — в `NOTICE`.
