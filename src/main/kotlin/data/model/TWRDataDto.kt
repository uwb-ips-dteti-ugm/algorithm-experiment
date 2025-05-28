package com.rizqi.data.model

import com.rizqi.domain.model.TWRData
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
data class TWRDataDto(
    @SerialName("timestamp") val timestamp: Long,
    @SerialName("addr_1") val addr1: Int,
    @SerialName("addr_2") val addr2: Int,
    @SerialName("distance") val distance: Double
)

fun TWRDataDto.toDomain() = TWRData(timestamp, addr1, addr2, distance)
fun TWRData.toDto() = TWRDataDto(timestamp, addr1, addr2, distance)

@Serializable
data class TWRDataWrapper(
    @SerialName("twr_data") val twrData: List<TWRDataDto>
)
