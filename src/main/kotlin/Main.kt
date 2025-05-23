package com.rizqi

import com.rizqi.data.remote.WebSocketClient
import com.rizqi.data.repository.TWRRepositoryImpl
import com.rizqi.domain.usecase.GetNewPositionUseCase

fun main() {
    val wsClient = WebSocketClient("ws://localhost:8080")
    val repository = TWRRepositoryImpl(wsClient)
    val getNewPositionUseCase = GetNewPositionUseCase(repository)

    getNewPositionUseCase.start()
}