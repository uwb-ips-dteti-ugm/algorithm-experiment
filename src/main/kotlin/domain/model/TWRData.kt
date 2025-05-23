package com.rizqi.domain.model

import com.squareup.moshi.JsonClass

@JsonClass(generateAdapter = true)
data class TWRData(
    val timestamp: Long,
    val addr1: Int,
    val addr2: Int,
    val distance: Double,
)
