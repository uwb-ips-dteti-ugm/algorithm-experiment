package com.rizqi.data.repository

import com.rizqi.data.remote.WebSocketClient
import com.rizqi.domain.model.TWRData
import com.rizqi.domain.repository.TWRRepository
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

class TWRRepositoryImpl(private val client: WebSocketClient): TWRRepository {
    override fun getTWRData(): Flow<List<TWRData>> {
        return client.service.observeMessages().map { it.twr_data }
    }

    override fun sendMessage(message: String) {
        client.service.sendMessage(message)
    }
}