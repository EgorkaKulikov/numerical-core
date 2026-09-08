plugins {
    kotlin("jvm") version "2.0.0"
    `java-library`
    `maven-publish`
    id("org.jetbrains.kotlinx.kover") version "0.8.3"
}

repositories {
    mavenCentral()
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
    // Режим `explicitApi()` НЕ включён сознательно: он потребовал бы `public` на каждом
    // из ~150 объявлений, то есть массовой косметической правки поверх структурного
    // рефакторинга. Поверхность API описана в README (раздел «Публичный API»);
    // внутренние детали (`DenseOps`, `Backends.resolve`) уже помечены `internal`.
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
}

/**
 * Быстрый набор (тег `fast`). В этой библиотеке все тесты быстрые, и `fastTest`
 * совпадает с `test` по составу; задача сохранена ради единого словаря команд с
 * соседними репозиториями (`minimal-splines`, `integral-equations`).
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
    systemProperty("numerics.backend", numericsBackend)
    outputs.upToDateWhen { false }
}

kover {
    currentProject {
        instrumentation {
            // Источник покрытия — `test`; `fastTest` дублирует его состав.
            disabledForTestTasks.add("fastTest")
            disabledForTestTasks.add("regenerateGolden")
        }
    }
}

// --- Публикация ---------------------------------------------------------------
// Локальная проверка: ./gradlew publishToMavenLocal
// Удалённый репозиторий НЕ задан намеренно: точка настройки — `repositories { maven { ... } }`
// ниже, адрес и учётные данные берутся из свойств Gradle/переменных окружения.
publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
            pom {
                name.set("numerical-core")
                description.set(
                    "Универсальные средства численных вычислений на Kotlin/JVM: составная квадратура " +
                        "Гаусса–Лежандра, плотная линейная алгебра с подключаемыми бэкендами, оценка " +
                        "обусловленности и прямой ошибки, параллельная сборка матриц.",
                )
                url.set("https://github.com/EgorkaKulikov/numerical-core")
                licenses {
                    license {
                        name.set("Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0")
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
        // Пример подключения удалённого репозитория (раскомментировать и задать свойства):
        // maven {
        //     name = "remote"
        //     url = uri(providers.gradleProperty("publishUrl").getOrElse(""))
        //     credentials(PasswordCredentials::class) // remoteUsername / remotePassword
        // }
    }
}

// Микробенчмарк публичного API (не входит в артефакт и в test).
val benchmark: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets["main"].output + configurations["runtimeClasspath"]
    runtimeClasspath += output + compileClasspath
}
tasks.register<JavaExec>("benchmark") {
    description = "Микробенчмарк публичного API; размеры матриц — через -Pbench.args=\"256 1024\""
    group = "verification"
    classpath = benchmark.runtimeClasspath
    mainClass.set("numerics.bench.BenchKt")
    maxHeapSize = "4g"
    args = (project.findProperty("bench.args") as String? ?: "256 1024").split(" ").filter { it.isNotBlank() }
    systemProperty("numerics.backend", numericsBackend)
}
