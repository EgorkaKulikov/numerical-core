# numerical-core

Универсальные средства численных вычислений на Kotlin/JVM: составная квадратура
Гаусса–Лежандра, плотная линейная алгебра с подключаемыми бэкендами, оценка
обусловленности и прямой ошибки решения СЛАУ, параллельная сборка матриц, порог
достоверности измеренных величин.

Библиотека не знает ни о сплайнах, ни об интегральных уравнениях: все её типы
работают с `Double`, `DoubleArray` и `Array<DoubleArray>`. Она служит нижним слоем
для библиотеки минимальных сплайнов и для прикладных проектов, построенных на ней.

```
numerical-core  <--  minimal-splines  <--  integral-equations
       ^                                          |
       +------------------------------------------+
```

Стрелки — направление зависимости. Обратных зависимостей нет: `numerical-core`
собирается и тестируется в одиночестве.

## Что входит

| Пакет | Сущность | Назначение |
|---|---|---|
| `numerics` | `GaussLegendre` | составная квадратура Гаусса–Лежандра по произвольному разбиению отрезка |
| `numerics` | `LinearAlgebra` | фасад плотной линейной алгебры (`solve`, `solveDiagnosed`, `matVec`, `matMat`, `atWa`, `cholesky`, нормы); делегирует активному бэкенду |
| `numerics` | `ReferenceLinearAlgebra` | эталонная реализация на чистом Kotlin — оракул для сверки бэкендов |
| `numerics.backend` | `LinAlgBackend`, `Backends`, `MultikCpuBackend`, `ReferenceBackend` | SPI бэкендов: нативный multik/OpenBLAS и чистый JVM с автоматическим откатом |
| `numerics` | `NumericsContext` | явный контекст «чем считать и как распараллеливать» вместо глобального состояния |
| `numerics` | `ParallelAssembly` | параллельная по строкам сборка матриц без гонок |
| `numerics` | `Conditioning`, `ConditionEstimate`, `ForwardError` | число обусловленности, обратная и прямая ошибка решения СЛАУ — с признаком достоверности |
| `numerics` | `Measured`, `measured`, `reliableOrders`, `reliableConstCh` | порог достоверности величин: ниже него число считается шумом округления, а не измерением |
| `numerics` | `orders`, `constCh` | порядок сходимости `log2(E_h/E_{h/2})` и константа `E_h/h^p` по готовой таблице погрешностей |

## Что намеренно НЕ входит

- **Минимальные сплайны**: сетки с кратными узлами, порождающие системы, базис,
  аппроксимационные функционалы, квазиинтерполяция, метрика `errorEh` — всё это в
  библиотеке `minimal-splines`.
- **Интегральные уравнения**: решатели Фредгольма, Вольтерры, Урысона, модельные
  задачи, характеризационные эталоны, сверка с публикациями — в проекте
  `integral-equations`.
- **Ввод-вывод и демонстрации**: библиотека ничего не печатает и не читает.

Если для использования этой библиотеки требуется импортировать что-либо из
`splines.*` или `solvers.*` — это ошибка архитектуры, о ней следует сообщить.

## Подключение

Нужен JDK 21+. Координаты артефакта: `io.github.egorkakulikov:numerical-core:0.1.0`.

```kotlin
// build.gradle.kts потребителя
repositories {
    mavenCentral()
    mavenLocal() // после ./gradlew publishToMavenLocal в этом репозитории
    // Удалённый репозиторий пока не назначен; точка настройки:
    // providers.gradleProperty("numericsRepositoryUrl").orNull?.let { maven(url = uri(it)) }
}

dependencies {
    implementation("io.github.egorkakulikov:numerical-core:0.1.0")
}
```

Локальная публикация из этого репозитория:

```bash
./gradlew publishToMavenLocal
ls ~/.m2/repository/io/github/egorkakulikov/numerical-core/0.1.0/
```

Артефакт содержит `numerical-core-0.1.0.jar`, `-sources.jar`, POM и Gradle module
metadata. Зависимости multik объявлены как `implementation`: типы multik не входят
в публичный API, потребителю они нужны только во время выполнения.

## Минимальный пример

```kotlin
import numerics.Conditioning
import numerics.ForwardError
import numerics.GaussLegendre
import numerics.LinearAlgebra
import kotlin.math.PI
import kotlin.math.sin

fun main() {
    // 1. Составная квадратура: интеграл sin(t) по [0, pi] с разбиением на 4 подынтервала.
    val quad = GaussLegendre(nodesPerSub = 8)
    val breakpoints = DoubleArray(5) { it * PI / 4 }
    val integral = quad.integrate(breakpoints) { t -> sin(t) }
    println("∫ sin = $integral (точно 2)")

    // 2. Плотная СЛАУ на активном бэкенде (multik/OpenBLAS или reference).
    val a = arrayOf(
        doubleArrayOf(4.0, 1.0, 0.0),
        doubleArrayOf(1.0, 4.0, 1.0),
        doubleArrayOf(0.0, 1.0, 4.0),
    )
    val b = doubleArrayOf(1.0, 2.0, 3.0)
    val x = LinearAlgebra.solve(a, b)
    println("x = ${x.toList()}")

    // 3. Диагностика: обратная ошибка, cond и граница прямой ошибки — с признаком достоверности.
    val d = LinearAlgebra.solveDiagnosed(a, b)
    when (val fe = d.forwardError) {
        is ForwardError.Bounded -> println("верных разрядов не меньше ${fe.survivingDigitsOrNull()}")
        is ForwardError.NoFiniteBound -> println("конечного cond у матрицы нет")
        is ForwardError.Unreliable -> println("оценка cond недостоверна: невязка обращения ${fe.condition.inversionResidual}")
    }

    // 4. Число обусловленности напрямую.
    val est = Conditioning.conditionInf(a)
    println(est.valueOrNull()?.let { "cond_inf = $it" } ?: "оценка недостоверна")
}
```

## Основные концепции API

**Квадратура.** `GaussLegendre(nodesPerSub)` хранит эталонные узлы и веса на `[-1, 1]`
(`refNodesWeights()` возвращает копии) и интегрирует по составному разбиению:
`integrate(breakpoints, f)` и `integrateInterval(lo, hi, f)`. С `m` узлами точна для
многочленов степени `2m - 1`.

**Линейная алгебра.** `LinearAlgebra` — фасад: каждая операция принимает
`backend: LinAlgBackend = Backends.default()`. `solve` проверяет невязку результата по
относительному порогу `LinearAlgebra.SINGULARITY_RELATIVE_TOLERANCE` и бросает
`IllegalStateException` на вырожденной или нечисловой системе — на любом бэкенде.
`solveDiagnosed` дополнительно возвращает `ForwardError` (см. [docs/ACCURACY.md](docs/ACCURACY.md)).

**Бэкенды.** `LinAlgBackend` — интерфейс тяжёлых операций (`matVec`, `matTransVec`,
`matMat`, `atWa`, `addScaled`, `solve`) плюс `name` и `isAvailable()`. `Backends.default()`
выбирает первый доступный из `MultikCpuBackend` (нативный OpenBLAS) и `ReferenceBackend`
(чистый JVM); выбор переопределяется системным свойством `-Dnumerics.backend=multik|reference`.
Подробности — [docs/HPC.md](docs/HPC.md).

**Контекст вычислений.** `NumericsContext(backend, parallel)` — неизменяемое значение,
которое передаётся в объекты, решающие СЛАУ или собирающие матрицы. `NumericsContext.default()`
— разделяемый экземпляр; `NumericsContext.requireSame(owner, expected, dependency, actual)`
громко отвергает несовпадение контекстов у частей одной задачи.

**Параллельная сборка.** `ParallelAssembly.assembleRows(rows, cols, parallel) { i -> row }`
и `assembleMatrix(rows, cols, parallel) { i, j -> cell }`. Результат побитово не зависит
от `parallel`: каждая задача пишет только в свою строку.

**Обусловленность и точность.** `Conditioning.conditionInf(a)` — оценка через обращение
с невязкой `‖A A⁻¹ − I‖∞` как признаком достоверности (`ConditionEstimate.isReliable`,
`valueOrNull()`); `Conditioning.conditionSymmetric(a)` — через спектр (вращения Якоби),
для симметричных матриц. `Conditioning.relativeBackwardError(a, b, x)` и
`forwardError(condition, backwardError)` дают закрытый тип `ForwardError` с тремя
режимами: `Bounded`, `NoFiniteBound`, `Unreliable`.

**Достоверность величин.** `measured(value, threshold)` классифицирует число как
`Measured.Reliable` или `Measured.AtNoiseLevel` (порог по умолчанию
`MACHINE_NOISE_THRESHOLD = 1e-13`); `ratio`, `orderOrNull`, `reliableOrders`,
`reliableConstCh` по построению возвращают `null`, если хотя бы один аргумент лежит на
уровне шума. `orders(errs)` и `constCh(eh, h, p)` — те же величины без порога.

### Публичный API и внутренние детали

Публичны все перечисленные выше типы и функции. Внутренними (`internal`) являются:

- `DenseOps` — дешёвые матричные операции, общие для фасада и эталона;
- `Backends.select(...)` — выбор бэкенда по запрошенному имени и предикату доступности
  (открыт только тестам модуля).

Библиотека версии `0.x` ещё не имеет внешних пользователей, поэтому совместимость
API до `1.0.0` не гарантируется; изменения фиксируются в сообщениях коммитов.

## Тесты

```bash
./gradlew test                              # весь набор (113 тестов, единицы секунд)
./gradlew fastTest                          # тег fast — в этой библиотеке совпадает с test
./gradlew test -Dnumerics.backend=reference # тот же набор на чистом JVM-бэкенде
./gradlew koverHtmlReport                   # покрытие: build/reports/kover/html/index.html
./gradlew check                             # test + отчёты Kover
```

Бэкенд в тестах по умолчанию — `multik`; при недоступности нативной библиотеки
`Backends` откатывается на `reference`, и `BackendSpiTest` сообщает об этом явно.

## Гарантии верификации

| Что проверяется | Тест |
|---|---|
| Эквивалентность нативного и эталонного бэкендов на всех операциях фасада (в допуске) | `LinearAlgebraVsReferenceTest` |
| Единая семантика вырожденности и нечислового входа; относительный порог, не зависящий от масштаба | `LinearAlgebraRegressionTest` |
| Контракт фасада: формы входа, отказ на пустых/неквадратных/рваных матрицах, перестановки строк в LU | `LinearAlgebraTest`, `LinearAlgebraExtraTest` |
| Точность квадратуры на многочленах степени `2m - 1`, согласие `integrate`/`integrateInterval`, составное разбиение | `QuadratureTest` |
| Инкапсуляция: `refNodesWeights()` отдаёт копии | `QuadratureDefensiveCopyTest` |
| Оценка `cond`: достоверность через невязку обращения, согласие с методом Якоби на симметричных матрицах | `ConditioningTest` |
| Три режима `ForwardError`, граница `cond · ω`, отказ печатать недостоверное число | `ForwardErrorTest` |
| Побитовая идентичность параллельной и последовательной сборки | `ParallelAssemblyTest`, `ParallelAssemblyToggleTest` |
| Выбор бэкенда, откат, системное свойство | `BackendSpiTest` |
| Порог достоверности: `null` вместо арифметики на шуме | `ReliabilityTest`, `ConvergenceRatesTest` |

Независимая сверка узлов Гаусса–Лежандра с NumPy и решения СЛАУ со SciPy выполняется в
проекте `integral-equations` (`ScipyCrossVerificationTest`, `tools/verify_with_scipy.py`):
там уже есть инструментарий выгрузки артефактов, и разрезать его между репозиториями
не имело бы смысла.

## Источники

Происхождение алгоритмов — в [docs/REFERENCES.md](docs/REFERENCES.md). Границы точности
и протокол диагностики — в [docs/ACCURACY.md](docs/ACCURACY.md), устройство бэкендов и
параллелизма — в [docs/HPC.md](docs/HPC.md).

## Разработка

Правила для людей и агентов — в [AGENTS.md](AGENTS.md). Кратко:

1. Ничего из `splines.*`, `solvers.*`, `problems.*` в этой библиотеке импортировать нельзя.
2. Новый алгоритм приходит с тестом и строкой в `docs/REFERENCES.md`.
3. Изменение формул проверяется на характеризационных эталонах проекта
   `integral-equations` против опубликованного артефакта — см. AGENTS.md, раздел
   «Численные инварианты и эталоны».
4. Перед отправкой изменений: `./gradlew check`.

Соседние репозитории: `minimal-splines` (зависит от этой библиотеки),
`integral-equations` (зависит от обеих). История кода до разделения — в репозитории
`Numerical-Algorithms`; план разделения сохранён в
[docs/REPOSITORY_SPLIT_PLAN.md](docs/REPOSITORY_SPLIT_PLAN.md).

## Лицензия

Apache License 2.0 — см. [LICENSE](LICENSE).
