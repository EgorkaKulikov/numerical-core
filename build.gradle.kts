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
    // Нативный бэкенд линейной алгебры (multik/OpenBLAS). Типы multik НЕ выходят
    // в публичный API (`MultikCpuBackend` реализует наш `LinAlgBackend`), поэтому
    // зависимость — `implementation`, а не `api`: потребителю она нужна лишь в runtime.
    implementation("org.jetbrains.kotlinx:multik-core:0.2.3")
    implementation("org.jetbrains.kotlinx:multik-default:0.2.3")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

kotlin {
    jvmToolchain(21)
    // Режим `explicitApi()` НЕ включён сознательно: он потребовал бы `public` на каждом
    // из ~150 объявлений, то есть массовой косметической правки поверх структурного
    // рефакторинга. Поверхность API описана в README (раздел «Публичный API»);
    // внутренние детали (`DenseOps`, `Backends.select`) уже помечены `internal`.
}

java {
    withSourcesJar()
}

// --- Бэкенд линейной алгебры в тестах -----------------------------------------
// `Backends.select` при недоступности нативной библиотеки МОЛЧА откатывается на
// `ReferenceBackend`. Явное значение делает выбор видимым; внешнее `-Dnumerics.backend=...`
// уважается и позволяет прогонять тесты на обоих бэкендах.
val numericsBackend: String = System.getProperty("numerics.backend") ?: "multik"

tasks.test {
    useJUnitPlatform()
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
    useJUnitPlatform { includeTags("fast") }
    systemProperty("numerics.backend", numericsBackend)
}

kover {
    currentProject {
        instrumentation {
            // Источник покрытия — `test`; `fastTest` дублирует его состав.
            disabledForTestTasks.add("fastTest")
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
