# numerical-core

Плотная линейная алгебра для JVM, которая работает со скоростью BLAS/LAPACK той машины, где
запущена, и говорит, на сколько знаков можно верить ответу.

## Зачем ещё одна библиотека

Готового на JVM хватает, но каждое средство закрывает лишь часть задачи. Биндинги к LAPACK (netlib,
bytedeco, MKL) отдают `dgesv` над плоскими массивами и возвращают `info = 0` — это значит лишь, что
разложение завершилось, а не что решение верно. Чистые Java-библиотеки (EJML, ojAlgo, Hipparchus) удобны,
но на матрицах в тысячи строк медленнее нативного LAPACK в 10–20 раз и не пользуются MKL или Accelerate
того узла, где запущены. И ни одна не говорит, сколько значащих цифр в полученном `x` настоящие.
numerical-core — тонкий слой, закрывающий именно этот разрыв.

**Достоверность закреплена в типах.** `solve` не отдаст решение, не проверив обратную ошибку:
вырожденная система — исключение, а не мусор. `solveDiagnosed` возвращает `ForwardError` с тремя
исходами — граница есть, границы нет, оценка недостоверна — и компилятор заставит разобрать все три.

**Скорость нативной библиотеки той машины, где запущено.** Матрица хранится в столбцовом порядке
LAPACK и уходит в `dgesv`/`dgemm` без переупорядочения. На узле с MKL будет MKL, на Apple — Accelerate,
на Linux — системная OpenBLAS; потоками управляет планировщик кластера. Решение системы 1024×1024 —
10 мс против 140 мс у чистой Java; число обусловленности — 27 мс против 24 с при обращении через n решений.

**Одинаковое поведение везде.** Одна проверка невязки, одни исключения, одни форматы результата на
MKL, OpenBLAS, Accelerate и на переносимой Java-реализации, когда нативной нет — о переходе на неё
сообщается явно, тихого отката на медленный путь не бывает.

**Примитивы, которых в LAPACK нет, — рядом и в том же стиле.** Составная квадратура Гаусса–Лежандра,
параллельная сборка матриц с побитово воспроизводимым результатом, наблюдаемый порядок сходимости
только по тем измерениям, которые выше шума округления.

## Пример

```kotlin
import numerics.*
import kotlin.math.*

fun main() {
    // Собрать матрицу и правую часть.
    val a = DenseMatrix.fromRows(arrayOf(
        doubleArrayOf( 2.0, -1.0,  0.0),
        doubleArrayOf(-1.0,  2.0, -1.0),
        doubleArrayOf( 0.0, -1.0,  2.0),
    ))
    val b = doubleArrayOf(1.0, 0.0, 1.0)
    // Решить: вырожденная или нечисловая система отвергается исключением.
    val x: DoubleArray = LinearAlgebra.solve(a, b)
    // Решить и узнать, на сколько знаков верить ответу.
    val d = LinearAlgebra.solveDiagnosed(a, b, source = ConditionSource.ESTIMATE)
    when (val e = d.forwardError) {
        is ForwardError.Bounded       -> println("‖x − x*‖/‖x*‖ ≤ ${e.relativeBound}, cond ≈ ${e.cond}")
        is ForwardError.NoFiniteBound -> println("система численно вырождена, ω = ${e.backwardError}")
        is ForwardError.Unreliable    -> println("обусловленность не определена достоверно")
    }
    // Число обусловленности: оценка LAPACK за O(n²); null, если ей нельзя верить.
    val cond: Double? = Conditioning.conditionEstimate(a).valueOrNull()
    // Проинтегрировать: ∫₀^π sin t dt = 2.
    val integral = GaussLegendre(8).integrate(doubleArrayOf(0.0, PI / 2, PI)) { t -> sin(t) }
}
```

## Подключение

JDK 21 или новее. Зависимость — `io.github.egorkakulikov:numerical-core:1.0.0` из GitHub Packages; для чтения
нужен токен с правом `read:packages` (`gpr.user`/`gpr.token` в `~/.gradle/gradle.properties` или переменные окружения):

```kotlin
maven {
    url = uri("https://maven.pkg.github.com/EgorkaKulikov/numerical-core")
    credentials {
        username = providers.gradleProperty("gpr.user").orNull ?: System.getenv("GITHUB_ACTOR")
        password = providers.gradleProperty("gpr.token").orNull ?: System.getenv("GITHUB_TOKEN")
    }
}
```

Если на машине нет BLAS/LAPACK, добавьте `io.github.egorkakulikov:numerical-core-openblas:1.0.0` и
вызовите `OpenBlas.install()` до первого обращения к библиотеке — она распакует OpenBLAS для текущей платформы.

## Что внутри

`DenseMatrix` — матрица в формате LAPACK, которую можно отдавать нативной библиотеке без копии.
`LinearAlgebra` — решить систему, умножить, взять норму, разложить по Холецкому. `Conditioning` — узнать,
сколько знаков в решении верны: число обусловленности, спектр симметричной матрицы, обратная и прямая
ошибка. `GaussLegendre` — проинтегрировать по произвольному разбиению отрезка. `ParallelAssembly` —
собрать матрицу на всех ядрах с тем же результатом, что и в один поток. `Measured` — отделить измеренное
от шума округления и взять порядок сходимости по достоверным точкам. API — `./gradlew :numerical-core:dokkaHtml`.

## Проверить самому

`./gradlew build` прогоняет тесты и порог покрытия; `-Dnumerics.backend=java|native` выбирает реализацию,
в CI прогоняются обе. Тесты — эталоны поведения на фиксированных входах, свойства на случайных матрицах
против двух независимых оракулов, граничные случаи, параллельные вызовы. Измерения производительности —
`./gradlew :numerical-core:benchmark -Pbench.args="256 1024"`. Подробности: `docs/ТОЧНОСТЬ.md` (пороги и их
обоснование), `docs/ПРОИЗВОДИТЕЛЬНОСТЬ.md` (путь данных, потоки, таблица измерений), `docs/ИСТОЧНИКИ.md` (алгоритмы).

## Лицензия

Apache License 2.0, правообладатель — Куликов Егор Константинович. Текст лицензии — в `LICENSE`,
уведомления о сторонних компонентах — в `NOTICE`.
