//package com.rizqi
//
//import java.io.File
//
//data class RowData(
//    val points: List<Point>,
//    val distances: List<Triple<Int, Int, Double>>
//)
//
//object CsvLoader {
//    fun loadPerRow(path: String): List<RowData> {
//        val lines = File(path).readLines().drop(1)
//        val rowDataList = mutableListOf<RowData>()
//
//        for (line in lines) {
//            val cols = line.split(",").map { it.toDouble() }
//
//            val x1 = cols[6]; val y1 = cols[7]
//            val x2 = cols[0]; val y2 = cols[1]
//            val x3 = cols[2]; val y3 = cols[3]
//            val x4 = cols[4]; val y4 = cols[5]
//
//            val d12 = cols[9]
//            val d13 = cols[11]
//            val d14 = cols[13]
//            val d23 = cols[15]
//            val d24 = cols[17]
//            val d34 = cols[19]
//
//            val points = listOf(Point(x1, y1), Point(x2, y2), Point(x3, y3), Point(x4, y4))
//            val distances = listOf(
//                Triple(0, 1, d12),
//                Triple(0, 2, d13),
//                Triple(0, 3, d14),
//                Triple(1, 2, d23),
//                Triple(1, 3, d24),
//                Triple(2, 3, d34),
//            )
//
//            rowDataList += RowData(points, distances)
//        }
//
//        return rowDataList
//    }
//}
