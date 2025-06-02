package com.rizqi

import java.io.File

object CsvLoader {
    fun load(path: String): Pair<List<DataPoint>, List<Point>> {
        val lines = File(path).readLines()
        val allPoints = mutableListOf<Point>()
        val pointToIndex = mutableMapOf<Point, Int>()
        val dataPoints = mutableListOf<DataPoint>()

        lines.drop(1).forEach { line ->
            val cols = line.split(",").map { it.toDouble() }

            val x1 = cols[6]
            val y1 = cols[7]
            val x2 = cols[0]
            val y2 = cols[1]
            val x3 = cols[2]
            val y3 = cols[3]
            val x4 = cols[4]
            val y4 = cols[5]

            val d12 = cols[9]
            val d13 = cols[11]
            val d14 = cols[13]
            val d23 = cols[15]
            val d24 = cols[17]
            val d34 = cols[19]

            val p1 = Point(x1, y1)
            val p2 = Point(x2, y2)
            val p3 = Point(x3, y3)
            val p4 = Point(x4, y4)

            // Register unique points and get their indices
            val i1 = pointToIndex.getOrPut(p1) {
                allPoints.add(p1)
                allPoints.lastIndex
            }
            val i2 = pointToIndex.getOrPut(p2) {
                allPoints.add(p2)
                allPoints.lastIndex
            }
            val i3 = pointToIndex.getOrPut(p3) {
                allPoints.add(p3)
                allPoints.lastIndex
            }
            val i4 = pointToIndex.getOrPut(p4) {
                allPoints.add(p4)
                allPoints.lastIndex
            }

            // Create DataPoints with indices
            dataPoints += listOf(
                DataPoint(i1, i2, d12),
                DataPoint(i1, i3, d13),
                DataPoint(i1, i4, d14),
                DataPoint(i2, i3, d23),
                DataPoint(i2, i4, d24),
                DataPoint(i3, i4, d34),
            )
        }

        return Pair(dataPoints, allPoints)
    }
}
