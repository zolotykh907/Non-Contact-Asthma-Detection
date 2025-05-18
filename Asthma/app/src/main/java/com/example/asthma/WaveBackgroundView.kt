package com.example.asthma

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.sin
import kotlin.random.Random

class WaveBackgroundView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

//    private var isRecording: Boolean = false
    private val circles = mutableListOf<Circle>()
    private val paint = Paint().apply {
        style = Paint.Style.STROKE // Только обводка
        strokeWidth = 2f
        isAntiAlias = true // Сглаживание
    }
    private var amplitude = 0f

    private data class Circle(
        var x: Float,
        var y: Float,
        var radius: Float,
        var alpha: Float,
        var time: Float = 0f
    ) {
        fun update(deltaTime: Float) {
            time += deltaTime
            radius += 50 * deltaTime
            alpha -= 0.3f * deltaTime
            radius += 10 * sin(time * 5f)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        val iterator = circles.iterator()
        while (iterator.hasNext()) {
            val circle = iterator.next()
            circle.update(0.05f)
            if (circle.alpha <= 0) {
                iterator.remove()
            } else {
                paint.color = Color.RED
                paint.alpha = (circle.alpha * 255).toInt()
                canvas.drawCircle(circle.x, circle.y, circle.radius, paint)
            }
        }

        if (amplitude > 0.01f && circles.size < 20 && Random.nextFloat() < 0.1f) {
            circles.add(
                Circle(
                    x = Random.nextFloat() * width,
                    y = Random.nextFloat() * height,
                    radius = 10f,
                    alpha = 0.8f
                )
            )
        }

        invalidate() // Постоянная перерисовка
    }

    fun updateAmplitude(newAmplitude: Float) {
        amplitude = newAmplitude
    }

    /** Включить запись (включает генерацию кругов) */
//    fun startRecording() {
//        isRecording = true
//    }
//
//    /** Выключить запись (останавливает генерацию новых кругов) */
//    fun stopRecording() {
//        isRecording = false
//    }
//
//    /** Проверка состояния записи */
//    fun isRecording(): Boolean = isRecording
}
