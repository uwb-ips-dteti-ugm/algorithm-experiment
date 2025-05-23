package com.rizqi.data.remote

import com.rizqi.domain.model.TWRData
import com.squareup.moshi.JsonClass

@JsonClass(generateAdapter = true)
data class TWRPayload(val twr_data: List<TWRData>)
