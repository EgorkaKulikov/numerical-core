import kotlinx.kover.gradle.plugin.dsl.CoverageUnit

plugins {
    kotlin("jvm")
    `java-library`
    `maven-publish`
    id("org.jetbrains.kotlinx.kover")
    id("org.jetbrains.dokka")
}

dependencies {
    // Линейная алгебра через BLAS/LAPACK (netlib): нативная реализация берётся из системы
    // (Accelerate, OpenBLAS, MKL), при её отсутствии используется переносимая реализация на Java.
    // Типы netlib не выходят в публичный API (`NetlibBackend` реализует наш `LinAlgBackend`),
    // поэтому зависимость — `implementation`, а не `api`: потребителю она нужна лишь в runtime.
    implementation("dev.ludovic.netlib:blas:3.2.0")
    implementation("dev.ludovic.netlib:lapack:3.2.0")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    // Property-based тесты (jqwik работает на JUnit Platform, теги транслируются в теги платформы)
    testImplementation("net.jqwik:jqwik:1.9.2")
    // Независимый оракул для линейной алгебры и квадратур Гаусса–Лежандра
    testImplementation("org.hipparchus:hipparchus-core:4.0.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

kotlin {
    jvmToolchain(21)
    // Явная видимость и явные типы у всех публичных объявлений: поверхность API
    // фиксируется в исходном коде, а не по умолчанию компилятора.
    explicitApi()
}

java {
    withSourcesJar()
}

// --- Реализация линейной алгебры в тестах -------------------------------------
// Свойство `numerics.backend` выбирает реализацию BLAS/LAPACK: `native` — только
// системная библиотека (ошибка, если не загрузилась), `java` — только переносимая
// реализация на Java, `auto` — нативная при доступности, иначе Java с предупреждением.
// Значение берётся из `-Pnumerics.backend=...` либо `-Dnumerics.backend=...`, по
// умолчанию `auto`; так один и тот же набор тестов прогоняется на обеих реализациях.
val numericsBackend: String =
    (project.findProperty("numerics.backend") as String?)
        ?: System.getProperty("numerics.backend")
        ?: "auto"

tasks.test {
    useJUnitPlatform { excludeTags("golden-generate") }
    systemProperty("numerics.backend", numericsBackend)
    // Проброс переключателей netlib в тестовую JVM: `-Ddev.ludovic.netlib.{blas,lapack}.allowNative=false`
    // принудительно включает F2J и позволяет локально воспроизвести CI без системной BLAS/LAPACK.
    listOf("dev.ludovic.netlib.lapack.allowNative", "dev.ludovic.netlib.blas.allowNative").forEach { key ->
        System.getProperty(key)?.let { systemProperty(key, it) }
    }
    // Многопоточные LU-разложения системного OpenBLAS используют стек вызывающего
    // потока; стандартного размера может не хватить.
    jvmArgs("-Xss8m")
}

/**
 * Быстрый набор (тег `fast`). В этой библиотеке все тесты быстрые, и `fastTest`
 * совпадает с `test` по составу; задача сохранена как отдельная точка запуска
 * подмножества тестов, помеченных тегом.
 */
tasks.register<Test>("fastTest") {
    group = "verification"
    description = "Быстрый набор тестов (тег fast)"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("fast")
        excludeTags("golden-generate")
    }
    systemProperty("numerics.backend", numericsBackend)
    // Проброс переключателей netlib в тестовую JVM: `-Ddev.ludovic.netlib.{blas,lapack}.allowNative=false`
    // принудительно включает F2J и позволяет локально воспроизвести CI без системной BLAS/LAPACK.
    listOf("dev.ludovic.netlib.lapack.allowNative", "dev.ludovic.netlib.blas.allowNative").forEach { key ->
        System.getProperty(key)?.let { systemProperty(key, it) }
    }
    // Многопоточные LU-разложения системного OpenBLAS используют стек вызывающего
    // потока; стандартного размера может не хватить.
    jvmArgs("-Xss8m")
}

/**
 * Перегенерация golden-эталонов поведения (JSON-файлы в `src/test/resources/golden`).
 * Тег `golden-generate` исключён из `test`/`fastTest`; запускать только при намеренном
 * изменении поведения. Подробности — в `src/test/resources/golden/README.md`.
 */
tasks.register<Test>("regenerateGolden") {
    description = "Перегенерировать эталоны поведения в src/test/resources/golden"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("golden-generate") }
    systemProperty("golden.dir", layout.projectDirectory.dir("src/test/resources/golden").asFile.absolutePath)
    systemProperty("golden.version", project.version.toString())
    systemProperty("numerics.backend", numericsBackend)
    // Многопоточные LU-разложения системного OpenBLAS используют стек вызывающего
    // потока; стандартного размера может не хватить.
    jvmArgs("-Xss8m")
    outputs.upToDateWhen { false }
}

kover {
    currentProject {
        instrumentation {
            // Источник покрытия — `test`; `fastTest` дублирует его состав.
            disabledForTestTasks.add("fastTest")
            disabledForTestTasks.add("regenerateGolden")
        }
        sources {
            // Измерения производительности — не библиотечный код, в покрытии не участвуют.
            excludedSourceSets.add("benchmark")
        }
    }
    reports {
        filters {
            excludes {
                packages("numerics.bench")
            }
        }
        // Планка покрытия: проверяется задачей `koverVerify`, входящей в `check`.
        // Замер при обеих доступных реализациях: строки 91.9 %, ветви 85.6 %;
        // порог — фактическое значение минус 2 %, чтобы результат не зависел от машины.
        verify {
            rule("Покрытие строк") {
                minBound(97)
            }
            rule("Покрытие ветвей") {
                bound {
                    minValue = 93
                    coverageUnits = CoverageUnit.BRANCH
                }
            }
        }
    }
}

tasks.check {
    dependsOn("koverVerify")
}

// --- Документация -------------------------------------------------------------
// HTML-документация публичного API: ./gradlew :numerical-core:dokkaHtml
// (результат в build/dokka/html). Публичные символы без KDoc выводятся предупреждениями.
tasks.dokkaHtml {
    moduleName.set("numerical-core")
    dokkaSourceSets.configureEach {
        includeNonPublic.set(false)
        reportUndocumented.set(true)
        jdkVersion.set(21)
    }
}

// --- Публикация ---------------------------------------------------------------
// Локальная проверка: ./gradlew publishToMavenLocal
// Удалённый репозиторий — GitHub Packages. Учётные данные берутся из переменных
// окружения GITHUB_ACTOR/GITHUB_TOKEN (CI) или свойств Gradle gpr.user/gpr.token;
// при их отсутствии publishToMavenLocal работает как прежде.
publishing {
    publications {
        create<MavenPublication>("maven") {
            artifactId = "numerical-core"
            from(components["java"])
            pom {
                name.set("numerical-core")
                description.set(
                    "Численные примитивы для плотных вычислений на Kotlin/JVM поверх BLAS/LAPACK " +
                        "целевой системы: решение систем линейных уравнений с оценкой достоверности " +
                        "результата, обусловленность, спектр симметричных матриц, разложение Холецкого, " +
                        "составная квадратура Гаусса–Лежандра, параллельная сборка матриц.",
                )
                url.set("https://github.com/EgorkaKulikov/numerical-core")
                inceptionYear.set("2026")
                licenses {
                    license {
                        name.set("Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0")
                        distribution.set("repo")
                    }
                }
                developers {
                    developer {
                        id.set("EgorkaKulikov")
                        name.set("Egor Kulikov")
                    }
                }
                scm {
                    url.set("https://github.com/EgorkaKulikov/numerical-core")
                }
            }
        }
    }
    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/EgorkaKulikov/numerical-core")
            credentials {
                username = providers.environmentVariable("GITHUB_ACTOR").orNull ?: providers.gradleProperty("gpr.user").orNull
                password = providers.environmentVariable("GITHUB_TOKEN").orNull ?: providers.gradleProperty("gpr.token").orNull
            }
        }
    }
}

// Измерения производительности публичного API (не входят в артефакт и в test).
val benchmark: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets["main"].output + configurations["runtimeClasspath"]
    runtimeClasspath += output + compileClasspath
}
tasks.register<JavaExec>("benchmark") {
    description = "Измерения производительности публичного API; размеры матриц — через -Pbench.args=\"256 1024\""
    group = "verification"
    classpath = benchmark.runtimeClasspath
    mainClass.set("numerics.bench.BenchKt")
    maxHeapSize = "4g"
    args = (project.findProperty("bench.args") as String? ?: "256 1024").split(" ").filter { it.isNotBlank() }
    systemProperty("numerics.backend", numericsBackend)
}
