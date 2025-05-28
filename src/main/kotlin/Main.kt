package com.rizqi

import com.rizqi.di.appModule
import com.rizqi.presentation.UWBViewModel
import kotlinx.coroutines.runBlocking
import org.koin.core.context.GlobalContext
import org.koin.core.context.startKoin

fun main() = runBlocking{
    val baseUrl = "ws://localhost:8080"

    startKoin {
        modules(appModule(baseUrl))
    }

    val viewModel: UWBViewModel = GlobalContext.get().get<UWBViewModel>()

    while (true){
        viewModel.startListening { data ->
            println("Received: $data")
        }
        Thread.sleep(1000) // 1s
    }

//    viewModel.stop()
}

