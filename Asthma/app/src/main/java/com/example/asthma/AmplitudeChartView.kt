package com.example.asthma

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

class AmplitudeChartView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val amplitudes = mutableListOf<Float>()
    private val linePaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private var isRecording = false

    fun startRecording() {
        isRecording = true
        amplitudes.clear() // очищаем при начале новой записи
        invalidate()
    }

    fun stopRecording() {
        isRecording = false
        invalidate()
    }

    fun updateAmplitude(newAmplitude: Float) {
        if (amplitudes.size >= 50) {
            amplitudes.removeAt(0) // удаляем старое
        }
        amplitudes.add(newAmplitude)
        invalidate()

    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (amplitudes.size < 2) return

        val maxVal = amplitudes.maxOrNull() ?: 1f
        val minVal = amplitudes.minOrNull() ?: 0f
        val range = maxVal - minVal

        val padding = 20f
        val chartWidth = width - padding * 2
        val chartHeight = height - padding * 2

        val stepX = chartWidth / (amplitudes.size - 1)

        val points = amplitudes.mapIndexed { index, amp ->
            val x = padding + index * stepX
            val y = padding + chartHeight * (1 - (amp - minVal) / (range.takeIf { it > 0 } ?: 1f))
            x to y
        }

        for (i in 0 until points.size - 1) {
            val (x1, y1) = points[i]
            val (x2, y2) = points[i + 1]
            canvas.drawLine(x1, y1, x2, y2, linePaint)
        }
    }
}
