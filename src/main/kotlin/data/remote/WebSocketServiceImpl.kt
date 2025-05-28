package com.rizqi.data.remote

import com.rizqi.data.model.TWRDataWrapper
import com.rizqi.data.model.toDomain
import com.rizqi.data.model.toDto
import com.rizqi.domain.model.TWRData
import io.ktor.client.*
import io.ktor.client.plugins.websocket.*
import io.ktor.websocket.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.Json

class WebSocketServiceImpl(
    private val baseUrl: String,
    private val client: HttpClient
) : WebSocketService {

    private var session: WebSocketSession? = null
    private var job: Job? = null

    override suspend fun connect(onDataReceived: (List<TWRData>) -> Unit) {
        if (session != null) return // Already connected

        println("Connecting to WebSocket: $baseUrl")

        try {
            session = client.webSocketSession(urlString = baseUrl)
            if (session?.isActive != true) {
                println("Failed to connect: WebSocket session not active")
                return
            }

            // Launch a long-lived coroutine to keep listening
            job = CoroutineScope(Dispatchers.IO).launch {
                try {
                    for (frame in session!!.incoming) {
                        if (frame is Frame.Text) {
                            val json = frame.readText()
                            val parsed = Json.decodeFromString(
                                TWRDataWrapper.serializer(),
                                json
                            )
                            onDataReceived(parsed.twrData.map { it.toDomain() })
                        }
                    }
                } catch (e: Exception) {
                    println("WebSocket listening error: ${e.localizedMessage}")
                } finally {
                    println("WebSocket listener ended.")
                    session?.close()
                    session = null
                }
            }

            println("WebSocket connected and listening")

        } catch (e: Exception) {
            println("WebSocket connection error: ${e.localizedMessage}")
        }
    }

    override suspend fun disconnect() {
        println("Disconnecting WebSocket...")
        try {
            job?.cancelAndJoin()
            job = null
            session?.close()
            session = null
            println("WebSocket disconnected.")
        } catch (e: Exception) {
            println("Error while disconnecting: ${e.localizedMessage}")
        }
    }

    override suspend fun send(data: TWRData) {
        val currentSession = session ?: throw IllegalStateException("WebSocket is not connected.")
        try {
            val payload = Json.encodeToString(
                TWRDataWrapper.serializer(),
                TWRDataWrapper(listOf(data.toDto()))
            )
            currentSession.send(Frame.Text(payload))
            println("Data sent via WebSocket.")
        } catch (e: Exception) {
            println("WebSocket send error: ${e.localizedMessage}")
        }
    }
}
