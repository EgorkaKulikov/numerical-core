plugins {
    kotlin("jvm") version "2.0.0" apply false
    id("org.jetbrains.kotlinx.kover") version "0.8.3" apply false
    id("org.jetbrains.dokka") version "1.9.20" apply false
}

allprojects {
    repositories { mavenCentral() }
}
