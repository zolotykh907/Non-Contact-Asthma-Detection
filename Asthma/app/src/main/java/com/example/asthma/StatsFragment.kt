package com.example.asthma

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment

class StatisticsFragment : Fragment() {

    private lateinit var amplitudeChartView: AmplitudeChartView
    private val handler = Handler(Looper.getMainLooper())
    private val updateInterval: Long = 100

    private val updateRunnable = object : Runnable {
        override fun run() {
            val newAmplitude = generateMockAmplitude()
            amplitudeChartView.updateAmplitude(newAmplitude)
            handler.postDelayed(this, updateInterval)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_stats, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        amplitudeChartView = view.findViewById(R.id.amplitudeChartView)
        handler.post(updateRunnable)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        handler.removeCallbacks(updateRunnable)
    }

    private fun generateMockAmplitude(): Float {
        return (0..100).random() / 100f
    }
}

