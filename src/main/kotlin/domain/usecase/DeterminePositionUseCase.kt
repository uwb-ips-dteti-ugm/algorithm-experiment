package com.rizqi.domain.usecase

import com.rizqi.domain.model.TWRData

class DeterminePositionUseCase {
    fun filterSymmetricPairs(data: List<TWRData>): List<TWRData> {
        val seenPairs = mutableSetOf<Pair<Int, Int>>()
        val result = mutableListOf<TWRData>()

        for (item in data) {
            val key = if (item.addr1 < item.addr2) {
                item.addr1 to item.addr2
            } else {
                item.addr2 to item.addr1
            }

            if (key !in seenPairs) {
                seenPairs.add(key)
                result.add(item)
            }
        }

        return result
    }

    fun mapToCircleCoordinate(data: List<TWRData>){

    }
}