plugins {
    kotlin("jvm") version "1.9.10"
    kotlin("kapt") version "1.9.10"
}

group = "com.rizqi"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("com.tinder.scarlet:scarlet:0.1.12")
    implementation("com.tinder.scarlet:websocket-okhttp:0.1.12")
    implementation("com.tinder.scarlet:message-adapter-moshi:0.1.12")
    implementation("com.tinder.scarlet:stream-adapter-coroutines:0.1.12")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.squareup.moshi:moshi:1.15.0")
    implementation("com.squareup.moshi:moshi-kotlin:1.15.0")
    kapt("com.squareup.moshi:moshi-kotlin-codegen:1.15.0")
    implementation("com.squareup.okhttp3:okhttp:4.11.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.11.0")
    implementation("com.squareup.okhttp3:logging-interceptor:4.11.0")
    implementation("com.tinder.scarlet:stream-adapter-rxjava2:0.1.12")
    implementation("io.reactivex.rxjava2:rxjava:2.2.21")

    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
}

kotlin {
    jvmToolchain(17)
}
