package com.rizqi.domain.model

data class CircleEquation(
    val x1 : Double,
    val x2: Double,
    val y1: Double,
    val y2: Double,
    val radius: Double,
) {
    fun getSimpleEquationNotation(): String {
        val equation = "C(${x1},${x2}):    (x${x2} - x${x1})^2 + (y${y2} - y${y1})^2 = d(${x1},${x2})^2"
        return equation
    }

    fun getGeneralEquationNotation(): String {
        val equation = "(x${x2} - x${x1})^2 + (y${y2} - y${y1})^2 + (d0,${x1})^2 - 2(d0,${x1})root(x${x1}^2 + y${x1}^2)(e0,${x1}^2)"
        return equation
    }
}
