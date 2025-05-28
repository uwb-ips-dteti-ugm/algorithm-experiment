package com.rizqi.presentation


import com.rizqi.domain.model.TWRData
import com.rizqi.domain.repository.UWBRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class UWBViewModel(private val repository: UWBRepository) {

    private val scope = CoroutineScope(Dispatchers.Default)

    fun startListening(onData: (List<TWRData>) -> Unit) {
        scope.launch {
            repository.connect(onData)
            // Optional: Add logic to handle onData callbacks
        }
    }

    fun sendTWRData(data: TWRData) {
        scope.launch {
            repository.sendData(data)
        }
    }

    fun stop() {
        scope.launch {
            repository.disconnect()
        }
    }
}
