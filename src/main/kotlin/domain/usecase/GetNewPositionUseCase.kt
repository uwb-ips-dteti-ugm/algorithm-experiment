package com.rizqi.domain.usecase

import com.rizqi.domain.model.TWRData
import com.rizqi.domain.repository.TWRRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class GetNewPositionUseCase(private val repository: TWRRepository){
    private val _data = MutableStateFlow<List<TWRData>>(emptyList())
    val data: StateFlow<List<TWRData>> = _data.asStateFlow()

    fun start(){
        CoroutineScope(Dispatchers.IO).launch {
            repository.getTWRData().collectLatest {
                _data.value = it
                println("Received: $it")
            }
        }
    }
}