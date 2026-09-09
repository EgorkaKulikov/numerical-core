plugins {
    kotlin("jvm")
    `java-library`
    `maven-publish`
}

kotlin {
    jvmToolchain(21)
    explicitApi()
}

dependencies {
    api(project(":numerical-core"))
    implementation("org.bytedeco:openblas-platform:0.3.34-1.5.14")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

java {
    withSourcesJar()
}

tasks.test {
    useJUnitPlatform()
    // Многопоточное LU-разложение в OpenBLAS требует стека больше стандартного (см. README модуля).
    jvmArgs("-Xss8m")
}

publishing {
    publications {
        create<MavenPublication>("maven") {
            artifactId = "numerical-core-openblas"
            from(components["java"])
            pom {
                name.set("numerical-core-openblas")
                description.set(
                    "Упакованная реализация BLAS/LAPACK (OpenBLAS) для numerical-core: " +
                        "используется там, где системной библиотеки нет"
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
