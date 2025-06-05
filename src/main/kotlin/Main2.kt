import org.apache.commons.math3.linear.Array2DRowRealMatrix
import org.apache.commons.math3.linear.ArrayRealVector
import org.apache.commons.math3.linear.QRDecomposition
import java.util.UUID
import kotlin.math.sqrt

fun main() {
    val x0 = Variable(id = UUID.randomUUID(), value = 0.0)
    val x1 = Variable(id = UUID.randomUUID(), value = 1.113437)
    val x2 = Variable(id = UUID.randomUUID(), value = 1.279734)
    val x3 = Variable(id = UUID.randomUUID(), value = 0.684363)

    val y0 = Variable(id = UUID.randomUUID(), value = 0.0)
    val y1 = Variable(id = UUID.randomUUID(), value = 0.804062)
    val y2 = Variable(id = UUID.randomUUID(), value = 1.983131)
    val y3 = Variable(id = UUID.randomUUID(), value = 0.344626)

    val point0 = Point(UUID.randomUUID(), x0, y0)
    val point1 = Point(UUID.randomUUID(), x1, y1)
    val point2 = Point(UUID.randomUUID(), x2, y2)
    val point3 = Point(UUID.randomUUID(), x3, y3)
    val points = listOf(point0, point1, point2, point3)

    val initPoint0 = Point(UUID.randomUUID(), x0.copy(value = 0.0), y0.copy(value = 0.0))
    val initPoint1 = Point(UUID.randomUUID(), x1.copy(value = 1.0), y1.copy(value = 0.9))
    val initPoint2 = Point(UUID.randomUUID(), x2.copy(value = 1.1), y2.copy(value = 1.8))
    val initPoint3 = Point(UUID.randomUUID(), x3.copy(value = 0.7), y3.copy(value = 0.4))

    val d01 = Distance(initPoint0, initPoint1, 1.375032
    )
    val d02 = Distance(initPoint0, initPoint2, 2.361733
    )
    val d03 = Distance(initPoint0, initPoint3, 0.773008
    )
    val d12 = Distance(initPoint1, initPoint2, 1.151819
    )
    val d13 = Distance(initPoint1, initPoint3, 0.541212
    )
    val d23 = Distance(initPoint2, initPoint3, 1.740796
    )
    val distances = listOf(d01, d02, d03, d12, d13, d23)

    val predictedPoints = newtonRaphson(distances, maxIter = 100, h = 1e-5)
    println()

    for (i in predictedPoints.indices){
        val predicted = predictedPoints[i]
        val actual = points.firstOrNull { it.x.id == predicted.x.id && it.y.id == predicted.y.id }

        println("Predicted -> x=${predicted.x.value}, y=${predicted.y.value}")
        println("Actual -> x=${actual?.x?.value}, y=${actual?.y?.value}")
        println()
    }
}

data class Variable(val id: UUID, var value: Double)
data class Point(
    val id: UUID,
    val x: Variable,
    val y: Variable,
)
data class Distance(
    var point1: Point,
    var point2: Point,
    val d: Double,
)

fun f(distances: List<Distance>): Double {distances
    var value = 0.0

    for (i in distances.indices) {
        val distance = distances[i]
        val point1 = distance.point1
        val point2 = distance.point2
        val d = distance.d

        val x1 = point1.x.value
        val x2 = point2.x.value
        val y1 = point1.y.value
        val y2 = point2.y.value

        val dx = x2 - x1
        val dy = y2 - y1
        val dist = sqrt(dx * dx + dy * dy)

        val error = dx * dx + dy * dy + d * d - 2 * d * dist
        value += error
    }

    return value
}

fun numericalGradient(
    variables: List<Variable>,
    distances: List<Distance>,
    h: Double = 1e-5,
): DoubleArray {
    println("Variables: ${variables.map { it.value }}")

    val grad = DoubleArray(variables.size)
    val variableMap = variables.associateBy { it.id }

    for (i in variables.indices) {
        val variable = variables[i]

        val distances1 = distances.deepCopyWithSharedVariables(variableMap)
        val distances2 = distances.deepCopyWithSharedVariables(variableMap)

        applyDeltaIfMatchToVariables(variableMap, variable.id, -h)
        val f1 = f(distances1)

        applyDeltaIfMatchToVariables(variableMap, variable.id, +2 * h)
        val f2 = f(distances2)

        grad[i] = (f2 - f1) / (2 * h)

        // Reset the variable value
        variable.value -= h
    }

    return grad
}

fun numericalHessian(
    variables: List<Variable>,
    distances: List<Distance>,
    h: Double = 1e-5,
): Array<DoubleArray> {
    val n = variables.size
    val hessian = Array(n) { DoubleArray(n) }
    val variableMap = variables.associateBy { it.id }

    for (i in 0 until n) {
        for (j in 0 until n) {
            val vi = variables[i]
            val vj = variables[j]

            // Create 4 shifted copies with shared variables
            val d1 = distances.deepCopyWithSharedVariables(variableMap)
            val d2 = distances.deepCopyWithSharedVariables(variableMap)
            val d3 = distances.deepCopyWithSharedVariables(variableMap)
            val d4 = distances.deepCopyWithSharedVariables(variableMap)

            // Apply deltas to variableMap (shared across copies)
            applyDeltaIfMatchToVariables(variableMap, vi.id, +h)
            applyDeltaIfMatchToVariables(variableMap, vj.id, +h)
            val f1 = f(d1)

            applyDeltaIfMatchToVariables(variableMap, vj.id, -2 * h)
            val f2 = f(d2)

            applyDeltaIfMatchToVariables(variableMap, vi.id, -2 * h)
            applyDeltaIfMatchToVariables(variableMap, vj.id, +2 * h)
            val f3 = f(d3)

            applyDeltaIfMatchToVariables(variableMap, vj.id, -2 * h)
            val f4 = f(d4)

            hessian[i][j] = (f1 - f2 - f3 + f4) / (4 * h * h)

            // Reset both variable values
            variableMap[vi.id]?.value = vi.value
            variableMap[vj.id]?.value = vj.value
        }
    }

    return hessian
}

fun newtonRaphson(
    distances: List<Distance>,
    maxIter: Int = 10,
    tolerance: Double = 1e-12,
    h: Double = 1e-5
): List<Point>{
    val copyOfDistances = distances.deepCopy()
    val points = extractUniquePoints(copyOfDistances).toList()
    val variables = extractUniqueVariables(copyOfDistances).toList()

    for (iter in 0 until maxIter){
        println("\n--- Iteration ${iter + 1} ---")

        val grad = numericalGradient(variables, copyOfDistances, h)
        val hessian = numericalHessian(variables, copyOfDistances, h)

        val gradVector = ArrayRealVector(grad)
        val hessianMatrix = Array2DRowRealMatrix(hessian)

        val deltas = try {
            QRDecomposition(hessianMatrix).solver.solve(gradVector)
        } catch (e: Exception){
            println("Hessian is singular at iteration $iter")
            break
        }

        for (i in variables.indices){
            val variable = variables[i]
            val delta = deltas.getEntry(i)
            variable.value -= delta
            points.forEach {
                if (it.x.id == variable.id){
                    it.x.value -= delta
                }
                if (it.y.id == variable.id){
                    it.y.value -= delta
                }
            }
        }

        copyOfDistances.forEach { distance ->
            points.find { p -> p.id == distance.point1.id  }?.let { distance.point1 = it }
            points.find { p -> p.id == distance.point2.id  }?.let { distance.point2 = it }
        }

        points.forEachIndexed { i, point ->

            println("Point $i -> x=${point.x.value}, y=${point.y.value}")
        }

        if (deltas.norm < tolerance) {
            println("Converged at iteration $iter")
            break
        }

    }

    return points
}

fun applyDeltaIfMatch(distance: Distance, variableId: UUID, delta: Double) {
    listOf(distance.point1.x, distance.point1.y, distance.point2.x, distance.point2.y).forEach {
        if (it.id == variableId) {
            it.value += delta
        }
    }
}

fun extractUniqueVariables(distances: List<Distance>): List<Variable> {
    val map = mutableMapOf<UUID, Variable>()
    for (d in distances) {
        listOf(d.point1.x, d.point1.y, d.point2.x, d.point2.y).forEach {
            map[it.id] = it
        }
    }
    return map.values.toList()
}


fun extractUniquePoints(distances: List<Distance>): Set<Point> {
    return distances.flatMap { listOf(it.point1, it.point2) }.toSet()
}

fun List<Distance>.deepCopy(): List<Distance> {
    return this.map { d ->
        Distance(
            point1 = Point(
                id = d.point1.id,
                x = d.point1.x.copy(),
                y = d.point1.y.copy()
            ),
            point2 = Point(
                id = d.point2.id,
                x = d.point2.x.copy(),
                y = d.point2.y.copy()
            ),
            d = d.d
        )
    }.toMutableList()
}

fun List<Distance>.deepCopyWithSharedVariables(variableMap: Map<UUID, Variable>): List<Distance> {
    return this.map { d ->
        Distance(
            point1 = Point(
                id = d.point1.id,
                x = variableMap[d.point1.x.id] ?: error("Missing variable"),
                y = variableMap[d.point1.y.id] ?: error("Missing variable")
            ),
            point2 = Point(
                id = d.point2.id,
                x = variableMap[d.point2.x.id] ?: error("Missing variable"),
                y = variableMap[d.point2.y.id] ?: error("Missing variable")
            ),
            d = d.d
        )
    }
}

fun applyDeltaIfMatchToVariables(variables: Map<UUID, Variable>, id: UUID, delta: Double) {
    variables[id]?.value = variables[id]?.value?.plus(delta) ?: error("Variable not found")
}




