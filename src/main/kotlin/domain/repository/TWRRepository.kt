package com.rizqi.domain.repository

import com.rizqi.domain.model.TWRData
import kotlinx.coroutines.flow.Flow

interface TWRRepository {
    fun getTWRData(): Flow<List<TWRData>>
    fun sendMessage(message: String)
}