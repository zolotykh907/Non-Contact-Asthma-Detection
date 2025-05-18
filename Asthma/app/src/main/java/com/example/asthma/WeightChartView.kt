package com.example.asthma

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

class WeightChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    var weightData: List<Float> = emptyList()
        set(value) {
            field = value
            invalidate()
        }

    private val linePaint = Paint().apply {
        color = Color.BLUE
        strokeWidth = 6f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val pointPaint = Paint().apply {
        color = Color.RED
        strokeWidth = 10f
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val axisPaint = Paint().apply {
        color = Color.BLACK
        strokeWidth = 3f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.BLACK
        textSize = 30f
        isAntiAlias = true
    }

    private val notePaint = Paint().apply {
        color = Color.DKGRAY
        textSize = 28f
        isAntiAlias = true
        textAlign = Paint.Align.CENTER
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (weightData.size < 2) return

        val paddingLeft = 80f
        val paddingTop = 50f
        val paddingRight = 50f
        val paddingBottom = 100f

        val chartWidth = width - paddingLeft - paddingRight
        val chartHeight = height - paddingTop - paddingBottom

        val maxWeight = weightData.maxOrNull() ?: 1f
        val minWeight = weightData.minOrNull() ?: 0f
        val range = maxWeight - minWeight

        val stepX = chartWidth / (weightData.size - 1)
        val points = weightData.mapIndexed { index, weight ->
            val x = paddingLeft + index * stepX
            val y = paddingTop + chartHeight * (1 - (weight - minWeight) / (range.takeIf { it > 0 } ?: 1f))
            x to y
        }

        // Оси
        canvas.drawLine(paddingLeft, paddingTop, paddingLeft, paddingTop + chartHeight, axisPaint) // Y
        canvas.drawLine(paddingLeft, paddingTop + chartHeight, paddingLeft + chartWidth, paddingTop + chartHeight, axisPaint) // X

        // Подписи к оси Y
        canvas.drawText("${maxWeight} кг", 10f, paddingTop + 10f, textPaint)
        canvas.drawText("${minWeight} кг", 10f, paddingTop + chartHeight, textPaint)

        // Линии графика
        for (i in 0 until points.size - 1) {
            val (x1, y1) = points[i]
            val (x2, y2) = points[i + 1]
            canvas.drawLine(x1, y1, x2, y2, linePaint)
        }

        // Точки
        points.forEach { (x, y) ->
            canvas.drawCircle(x, y, 6f, pointPaint)
        }

        // Примечание под графиком
        canvas.drawText(
            "Контроль веса важен при астме: ожирение может усугубить симптомы.",
            width / 2f,
            height - 30f,
            notePaint
        )
    }
}
