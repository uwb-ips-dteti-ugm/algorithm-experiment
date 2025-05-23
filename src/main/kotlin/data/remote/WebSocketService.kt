package com.rizqi.data.remote

import com.tinder.scarlet.ws.Receive
import com.tinder.scarlet.ws.Send
import kotlinx.coroutines.flow.Flow

interface WebSocketService {
    @Receive
    fun observeMessages(): Flow<TWRPayload>

    @Send
    fun sendMessage(message: String)
}