package com.rizqi.domain.repository

import com.rizqi.domain.model.TWRData

interface UWBRepository {
    suspend fun connect(onDataReceived: (List<TWRData>) -> Unit)
    suspend fun disconnect()
    suspend fun sendData(data: TWRData)
}
