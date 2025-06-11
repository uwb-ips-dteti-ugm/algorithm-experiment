import kotlin.math.abs
import kotlin.math.pow
import kotlin.random.Random

fun generateShortId(length: Int = 8): String {
    val chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    return (1..length)
        .map { chars[Random.nextInt(chars.length)] }
        .joinToString("")
}

fun calculateError(previousValue: Double, currentValue: Double): Double {
    return if (previousValue != 0.0) {
        ((currentValue - previousValue) / previousValue) * 100
    } else {
        0.0
    }
}

data class Variable(
    val id: String,
    var value: Double,
)

data class Point(
    val id: String,
    var x: Variable,
    var y: Variable,
)

data class Distance(
    val id: String,
    var point1: Point,
    var point2: Point,
    val distance: Double,
)

fun main() {
//    Case: 2 Node [1 Server(0,0), 1 Client(x, 0)]
    val xActualServer = Variable(
        id = generateShortId(),
        value = 0.0,
    )
    val yActualServer = Variable(
        id = generateShortId(),
        value = 0.0,
    )
    val actualPointServer = Point(
        id = generateShortId(),
        x = xActualServer,
        y = yActualServer
    )

    val pointsKnowledge = mutableListOf(
        actualPointServer
    )

    val xActualClient1 = Variable(
        id = generateShortId(),
        value = 3.0,
    )
    val yActualClient1 = Variable(
        id = generateShortId(),
        value = 0.0,
    )
    val actualPointClient1 = Point(
        id = generateShortId(),
        x = xActualClient1,
        y = yActualClient1
    )

    val xActualClient2 = Variable(
        id = generateShortId(),
        value = 1.9,
    )
    val yActualClient2 = Variable(
        id = generateShortId(),
        value = 1.1,
    )
    val actualPointClient2 = Point(
        id = generateShortId(),
        x = xActualClient2,
        y = yActualClient2
    )

    val distance01 = Distance(
        id = generateShortId(),
        point1 = actualPointServer,
        point2 = actualPointClient1,
        distance = 3.0,
    )
    val distance02 = Distance(
        id = generateShortId(),
        point1 = actualPointServer,
        point2 = actualPointClient2,
        distance = 2.167,
    )
    val distance12 = Distance(
        id = generateShortId(),
        point1 = actualPointClient1,
        point2 = actualPointClient2,
        distance = 1.38,
    )
    val distances = listOf(distance01, distance02, distance12)
    val fixedPointIds = setOf(actualPointServer.id, actualPointClient1.id)
    val result = newtonRaphson(
        distances,
        fixedPointIds,
        h = 1e-10
    )
    println(result)
}

fun f(distance: Distance): Double {
    val x1 = distance.point1.x.value
    val x2 = distance.point2.x.value
    val y1 = distance.point1.y.value
    val y2 = distance.point2.y.value
    val d = distance.distance

    return (x2 - x1).pow(2.0) + (y2 - y1).pow(2.0) - d.pow(2.0)
}

fun createF(distances: List<Distance>) : List<Double>{
    return distances.map {
        f(it)
    }
}

fun calculateJacobian(
    distances: List<Distance>,
    variableIds: List<String>, // hanya variable yang boleh dihitung
    h: Double = 1e-5
): List<List<Double>> {
    val jacobian = mutableListOf<MutableList<Double>>()

    for (distance in distances) {
        val row = mutableListOf<Double>()

        for (varId in variableIds) {
            val value = when (varId) {
                distance.point1.x.id -> centralDifference(distance, { d, delta -> d.withPoint1X(d.point1.x.value + delta) }, h)
                distance.point2.x.id -> centralDifference(distance, { d, delta -> d.withPoint2X(d.point2.x.value + delta) }, h)
                distance.point1.y.id -> centralDifference(distance, { d, delta -> d.withPoint1Y(d.point1.y.value + delta) }, h)
                distance.point2.y.id -> centralDifference(distance, { d, delta -> d.withPoint2Y(d.point2.y.value + delta) }, h)
                else -> 0.0
            }
            row.add(value)
        }

        jacobian.add(row)
    }

    return jacobian
}

fun calculateDelta(jacobian: List<List<Double>>, fMatrix: List<Double>): List<Double> {
    val m = jacobian.size
    val n = jacobian[0].size

    // 1. transposeMatrix of Jacobian (n x m)
    val jT = transposeMatrix(jacobian)

    // 2. Jᵗ * J  => (n x n)
    val jtJ = multiply2DMatrix(jT, jacobian)

    // 3. Invert (Jᵗ * J)
    val jtJInv = invertMatrix(jtJ)

    // 4. Jᵗ * F  => (n x 1)
    val jtF = multiplyMatrix(jT, fMatrix)

    // 5. ΔX = - (Jᵗ * J)⁻¹ * (Jᵗ * F)
    return multiplyMatrix(jtJInv, jtF).map { -it }
}

fun newtonRaphson(
    initialDistances: List<Distance>,
    fixedPointIds: Set<String>,
    iteration: Int = 10,
    tolerance: Double = 1e-10,
    h: Double = 1e-5,
): List<Point> {
    var distances = initialDistances.toList()

    val points = distances
        .flatMap { listOf(it.point1, it.point2) }
        .distinctBy { it.id }
        .associateBy { it.id }
        .toMutableMap()

    val allVariables = distances.flatMap { d ->
        listOf(d.point1.x, d.point2.x, d.point1.y, d.point2.y)
    }.distinctBy { it.id }
        .associateBy { it.id }
        .toMutableMap()

    // Hanya ambil variabel dari point yang bukan fixed
    val variableIds = allVariables
        .filter { (id, variable) ->
            val parentPoint = points.values.find { it.x.id == id || it.y.id == id }
            parentPoint != null && parentPoint.id !in fixedPointIds
        }
        .map { it.key }

    repeat(iteration) {
        val fMatrix = createF(distances)
        val jacobian = calculateJacobian(distances, variableIds, h)
        val delta = calculateDelta(jacobian, fMatrix)

        if (delta.size != variableIds.size) {
            println("Delta size (${delta.size}) does not match variable size (${variableIds.size})")
            return@repeat
        }

        val maxDelta = delta.maxOf { abs(it) }
        if (maxDelta < tolerance) {
            println("Converged at iteration ${it + 1}")
            return@repeat
        }

        // Update only unfixed variables
        variableIds.forEachIndexed { index, varId ->
            val variable = allVariables[varId] ?: return@forEachIndexed
            allVariables[varId] = variable.copy(value = variable.value + delta[index])
        }

        // Update points with new variables
        points.values.forEach { point ->
            val updatedX = allVariables[point.x.id] ?: point.x
            val updatedY = allVariables[point.y.id] ?: point.y
            points[point.id] = point.copy(x = updatedX, y = updatedY)
        }

        // Update distances
        distances = distances.map { distance ->
            distance.copy(
                point1 = points[distance.point1.id] ?: distance.point1,
                point2 = points[distance.point2.id] ?: distance.point2
            )
        }
    }

    return points.values.toList()
}

// --- Extension helpers ---
fun Distance.withPoint1X(newValue: Double) = copy(point1 = point1.copy(x = point1.x.copy(value = newValue)))
fun Distance.withPoint2X(newValue: Double) = copy(point2 = point2.copy(x = point2.x.copy(value = newValue)))
fun Distance.withPoint1Y(newValue: Double) = copy(point1 = point1.copy(y = point1.y.copy(value = newValue)))
fun Distance.withPoint2Y(newValue: Double) = copy(point2 = point2.copy(y = point2.y.copy(value = newValue)))

// --- Central difference function ---
fun centralDifference(
    base: Distance,
    update: (Distance, Double) -> Distance,
    h: Double
): Double {
    val f1 = f(update(base, -h / 2))
    val f2 = f(update(base, +h / 2))
    return (f2 - f1) / h
}

fun transposeMatrix(matrix: List<List<Double>>): List<List<Double>> =
    List(matrix[0].size) { col -> List(matrix.size) { row -> matrix[row][col] } }

fun multiply2DMatrix(a: List<List<Double>>, b: List<List<Double>>): List<List<Double>> {
    val rowsA = a.size
    val colsA = a[0].size
    val colsB = b[0].size
    return List(rowsA) { i ->
        List(colsB) { j ->
            (0 until colsA).sumOf { k -> a[i][k] * b[k][j] }
        }
    }
}

fun multiplyMatrix(a: List<List<Double>>, b: List<Double>): List<Double> {
    return a.map { row -> row.zip(b).sumOf { (a, b) -> a * b } }
}

fun invertMatrix(matrix: List<List<Double>>): List<List<Double>> {
    val n = matrix.size
    val a = Array(n) { matrix[it].toDoubleArray() }
    val inv = Array(n) { DoubleArray(n) { if (it == it) 1.0 else 0.0 } }

    for (i in 0 until n) {
        var factor = a[i][i]
        if (factor == 0.0) throw ArithmeticException("Singular matrix")
        for (j in 0 until n) {
            a[i][j] /= factor
            inv[i][j] /= factor
        }
        for (k in 0 until n) {
            if (k != i) {
                factor = a[k][i]
                for (j in 0 until n) {
                    a[k][j] -= factor * a[i][j]
                    inv[k][j] -= factor * inv[i][j]
                }
            }
        }
    }
    return inv.map { it.toList() }
}


