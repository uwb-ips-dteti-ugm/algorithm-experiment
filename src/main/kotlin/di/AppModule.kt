package com.rizqi.di

import com.rizqi.data.remote.WebSocketService
import com.rizqi.data.remote.WebSocketServiceImpl
import com.rizqi.domain.model.TWRData
import com.rizqi.domain.repository.UWBRepository
import com.rizqi.presentation.UWBViewModel
import kotlinx.serialization.json.Json
import org.koin.dsl.module
import io.ktor.client.*
import io.ktor.client.engine.cio.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.plugins.websocket.WebSockets
import io.ktor.serialization.kotlinx.json.*

fun appModule(baseUrl: String) = module {

    single {
        HttpClient(CIO) {
            install(ContentNegotiation) {
                json(Json { ignoreUnknownKeys = true })
            }
        }
    }

    single<WebSocketService> {
        val client = HttpClient {
            install(WebSockets)
        }
        WebSocketServiceImpl(baseUrl, client)
    }

    single<UWBRepository> {
        object : UWBRepository {
            val service: WebSocketService = get()
            override suspend fun connect(onDataReceived: (List<TWRData>) -> Unit) = service.connect {
                onDataReceived(it)
            }
            override suspend fun disconnect() = service.disconnect()
            override suspend fun sendData(data: TWRData) = service.send(data)
        }
    }

    single { UWBViewModel(get()) }
}
