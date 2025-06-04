//package com.rizqi
//
//import org.apache.commons.math3.linear.Array2DRowRealMatrix
//import org.apache.commons.math3.linear.ArrayRealVector
//import org.apache.commons.math3.linear.LUDecomposition
//import java.io.File
//import kotlin.math.pow
//import kotlin.math.sqrt
//
//fun main() {
//    val path = "C:\\Users\\Lenovo\\OneDrive\\Dokumen\\dummy_points_dataset.csv"
//    val rows = CsvLoader.loadPerRow(path)
//
//    println("Starting optimization for ${rows.size} rows...\n")
//
//    rows.forEachIndexed { rowIndex, row ->
//        println("========== ROW ${rowIndex + 1} ==========")
//
//        val actualPoints = row.points
//        val distances = row.distances.map { DataPoint(it.first, it.second, it.third) }
//
//        val initialGuess = DoubleArray(8) { 0.5 }
//
//        val predictedVars = newtonRaphson(initialGuess, distances, maxIter = 100)
//
//        for (i in 0 until 4) {
//            val actual = actualPoints[i]
//            val xPred = predictedVars[2 * i]
//            val yPred = predictedVars[2 * i + 1]
//            val xErr = kotlin.math.abs(actual.x - xPred)
//            val yErr = kotlin.math.abs(actual.y - yPred)
//
//            println("Point ${i + 1}:")
//            println("  x_actual     = ${actual.x}")
//            println("  x_predicted  = $xPred")
//            println("  x_error      = $xErr")
//            println("  y_actual     = ${actual.y}")
//            println("  y_predicted  = $yPred")
//            println("  y_error      = $yErr")
//            println()
//        }
//
//        println("=========================================\n")
//    }
//}
//
//fun f(x: DoubleArray, dataPoints: List<DataPoint>): Double {
//    var value = 0.0
//    for (dp in dataPoints) {
//        val idx1 = dp.point1Index
//        val idx2 = dp.point2Index
//        val r = dp.distance
//
//        val x1 = x[2 * idx1]
//        val y1 = x[2 * idx1 + 1]
//        val x2 = x[2 * idx2]
//        val y2 = x[2 * idx2 + 1]
//
//        val dx = x2 - x1
//        val dy = y2 - y1
//        val dist = sqrt(dx * dx + dy * dy)
//
//        val error = dx * dx + dy * dy + r * r - 2 * r * dist
//        value += error
//    }
//    return value
//}
//
//fun numericalGradient(
//    f: (DoubleArray) -> Double,
//    x: DoubleArray,
//    h: Double = 1e-6
//): DoubleArray {
//    val grad = DoubleArray(x.size)
//    for (i in x.indices) {
//        val x1 = x.copyOf()
//        val x2 = x.copyOf()
//        x1[i] -= h
//        x2[i] += h
//        grad[i] = (f(x2) - f(x1)) / (2 * h)
//    }
//    return grad
//}
//
//fun numericalHessian(
//    f: (DoubleArray) -> Double,
//    x: DoubleArray,
//    h: Double = 1e-5
//): Array<DoubleArray> {
//    val n = x.size
//    val hessian = Array(n) { DoubleArray(n) }
//    for (i in 0 until n) {
//        for (j in 0 until n) {
//            val x1 = x.copyOf().apply { this[i] += h; this[j] += h }
//            val x2 = x.copyOf().apply { this[i] += h; this[j] -= h }
//            val x3 = x.copyOf().apply { this[i] -= h; this[j] += h }
//            val x4 = x.copyOf().apply { this[i] -= h; this[j] -= h }
//
//            hessian[i][j] = (f(x1) - f(x2) - f(x3) + f(x4)) / (4 * h * h)
//        }
//    }
//    return hessian
//}
//
//
//fun newtonRaphson(
//    initialX: DoubleArray,
//    dataPoints: List<DataPoint>,
//    tolerance: Double = 1e-6,
//    maxIter: Int = 10
//): DoubleArray {
//    var x = initialX.copyOf()
//
//    val fWithX: (DoubleArray) -> Double = { f(it, dataPoints) }
//
//    for (iter in 0 until maxIter) {
//        val grad = numericalGradient(fWithX, x)
//        val hess = numericalHessian(fWithX, x)
//
//        val gradVec = ArrayRealVector(grad)
//        val hessMat = Array2DRowRealMatrix(hess)
//
//        val delta = try {
//            LUDecomposition(hessMat).solver.solve(gradVec)
//        } catch (e: Exception) {
//            println("Hessian is singular at iteration $iter")
//            break
//        }
//
//        for (i in x.indices) {
//            x[i] -= delta.getEntry(i)
//        }
//
////        println("\n--- Iteration ${iter + 1} ---")
////        println("f(x): ${fWithX(x)}")
////        println("Gradient norm: ${gradVec.norm}")
////        println("Predicted coordinates:")
////        x.asList().chunked(2).forEachIndexed { i, (xPred, yPred) ->
////            println("  Point $i -> x=$xPred, y=$yPred")
////        }
//
//        if (delta.norm < tolerance) {
////            println("Converged at iteration $iter")
//            break
//        }
//    }
//
//    return x
//}
//
