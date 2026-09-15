import kotlinx.kover.gradle.plugin.dsl.CoverageUnit

plugins {
    kotlin("jvm")
    `java-library`
    `maven-publish`
    id("org.jetbrains.kotlinx.kover")
    id("org.jetbrains.dokka")
}

dependencies {
    // Linear algebra via BLAS/LAPACK (netlib): the native implementation is taken from the system
    // (Accelerate, OpenBLAS, MKL); when it is absent, the portable Java implementation is used.
    // netlib types do not leak into the public API (`NetlibBackend` implements our `LinAlgBackend`),
    // so the dependency is `implementation`, not `api`: consumers need it only at runtime.
    implementation("dev.ludovic.netlib:blas:3.2.0")
    implementation("dev.ludovic.netlib:lapack:3.2.0")

    testImplementation(kotlin("test"))
    testImplementation("org.junit.jupiter:junit-jupiter:5.10.2")
    // Property-based tests (jqwik runs on the JUnit Platform; its tags map to platform tags)
    testImplementation("net.jqwik:jqwik:1.9.2")
    // Independent oracle for linear algebra and Gauss–Legendre quadrature
    testImplementation("org.hipparchus:hipparchus-core:4.0.3")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

kotlin {
    jvmToolchain(21)
    // Explicit visibility and explicit types on all public declarations: the API surface
    // is fixed in the source code rather than by compiler defaults.
    explicitApi()
}

java {
    withSourcesJar()
}

// --- Linear algebra implementation in tests -----------------------------------
// The `numerics.backend` property selects the BLAS/LAPACK implementation: `native` — the
// system library only (an error if it fails to load), `java` — the portable Java
// implementation only, `auto` — native when available, otherwise Java with a warning.
// The value comes from `-Pnumerics.backend=...` or `-Dnumerics.backend=...`, defaulting
// to `auto`; this way the same test suite runs on both implementations.
val numericsBackend: String =
    (project.findProperty("numerics.backend") as String?)
        ?: System.getProperty("numerics.backend")
        ?: "auto"

tasks.test {
    useJUnitPlatform { excludeTags("golden-generate") }
    systemProperty("numerics.backend", numericsBackend)
    // Pass netlib switches through to the test JVM: `-Ddev.ludovic.netlib.{blas,lapack}.allowNative=false`
    // forces F2J and lets CI be reproduced locally without a system BLAS/LAPACK.
    listOf("dev.ludovic.netlib.lapack.allowNative", "dev.ludovic.netlib.blas.allowNative").forEach { key ->
        System.getProperty(key)?.let { systemProperty(key, it) }
    }
    // Multithreaded LU factorizations in the system OpenBLAS use the calling thread's
    // stack; the default size may be insufficient.
    jvmArgs("-Xss8m")
}

/**
 * Fast suite (tag `fast`). In this library all tests are fast, and `fastTest`
 * has the same contents as `test`; the task is kept as a separate entry point for
 * running the subset of tests marked with the tag.
 */
tasks.register<Test>("fastTest") {
    group = "verification"
    description = "Fast test suite (tag fast)"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform {
        includeTags("fast")
        excludeTags("golden-generate")
    }
    systemProperty("numerics.backend", numericsBackend)
    // Pass netlib switches through to the test JVM: `-Ddev.ludovic.netlib.{blas,lapack}.allowNative=false`
    // forces F2J and lets CI be reproduced locally without a system BLAS/LAPACK.
    listOf("dev.ludovic.netlib.lapack.allowNative", "dev.ludovic.netlib.blas.allowNative").forEach { key ->
        System.getProperty(key)?.let { systemProperty(key, it) }
    }
    // Multithreaded LU factorizations in the system OpenBLAS use the calling thread's
    // stack; the default size may be insufficient.
    jvmArgs("-Xss8m")
}

/**
 * Regenerates the golden behavior references (JSON files in `src/test/resources/golden`).
 * The `golden-generate` tag is excluded from `test`/`fastTest`; run only on a deliberate
 * behavior change. Details: `src/test/resources/golden/README.md`.
 */
tasks.register<Test>("regenerateGolden") {
    description = "Regenerate the golden behavior references in src/test/resources/golden"
    group = "verification"
    testClassesDirs = sourceSets["test"].output.classesDirs
    classpath = sourceSets["test"].runtimeClasspath
    useJUnitPlatform { includeTags("golden-generate") }
    systemProperty("golden.dir", layout.projectDirectory.dir("src/test/resources/golden").asFile.absolutePath)
    systemProperty("golden.version", project.version.toString())
    systemProperty("numerics.backend", numericsBackend)
    // Multithreaded LU factorizations in the system OpenBLAS use the calling thread's
    // stack; the default size may be insufficient.
    jvmArgs("-Xss8m")
    outputs.upToDateWhen { false }
}

kover {
    currentProject {
        instrumentation {
            // Coverage source is `test`; `fastTest` duplicates its contents.
            disabledForTestTasks.add("fastTest")
            disabledForTestTasks.add("regenerateGolden")
        }
        sources {
            // Performance measurements are not library code and do not count towards coverage.
            excludedSourceSets.add("benchmark")
        }
    }
    reports {
        filters {
            excludes {
                packages("numerics.bench")
            }
        }
        // Coverage bar: checked by the `koverVerify` task, which is part of `check`.
        // Measured on both available implementations: lines 91.9 %, branches 85.6 %;
        // the threshold is the actual value minus 2 %, so the result does not depend on the machine.
        verify {
            rule("Line coverage") {
                minBound(97)
            }
            rule("Branch coverage") {
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

// --- Documentation ------------------------------------------------------------
// HTML documentation of the public API: ./gradlew :numerical-core:dokkaHtml
// (output in build/dokka/html). Public symbols without KDoc are reported as warnings.
tasks.dokkaHtml {
    moduleName.set("numerical-core")
    dokkaSourceSets.configureEach {
        includeNonPublic.set(false)
        reportUndocumented.set(true)
        jdkVersion.set(21)
    }
}

// --- Publishing ---------------------------------------------------------------
// Local check: ./gradlew publishToMavenLocal
// Remote repository: GitHub Packages. Credentials come from the GITHUB_ACTOR/GITHUB_TOKEN
// environment variables (CI) or from the Gradle properties gpr.user/gpr.token;
// without them publishToMavenLocal works as before.
publishing {
    publications {
        create<MavenPublication>("maven") {
            artifactId = "numerical-core"
            from(components["java"])
            pom {
                name.set("numerical-core")
                description.set(
                    "Numerical primitives for dense computations on Kotlin/JVM on top of the target system's " +
                        "BLAS/LAPACK: linear system solves with a reliability estimate of the result, " +
                        "conditioning, spectra of symmetric matrices, Cholesky factorization, " +
                        "composite Gauss–Legendre quadrature, parallel matrix assembly.",
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

// Performance measurements of the public API (not part of the artifact or of test).
val benchmark: SourceSet by sourceSets.creating {
    compileClasspath += sourceSets["main"].output + configurations["runtimeClasspath"]
    runtimeClasspath += output + compileClasspath
}
tasks.register<JavaExec>("benchmark") {
    description = "Performance measurements of the public API; matrix sizes via -Pbench.args=\"256 1024\""
    group = "verification"
    classpath = benchmark.runtimeClasspath
    mainClass.set("numerics.bench.BenchKt")
    maxHeapSize = "4g"
    args = (project.findProperty("bench.args") as String? ?: "256 1024").split(" ").filter { it.isNotBlank() }
    systemProperty("numerics.backend", numericsBackend)
}
