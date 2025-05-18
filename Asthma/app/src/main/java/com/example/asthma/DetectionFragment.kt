package com.example.asthma

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.TextView
import androidx.core.app.ActivityCompat
import androidx.fragment.app.Fragment
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream

class DetectionFragment : Fragment() {

    private var isRecording = false
    private val SAMPLE_RATE = 16000
    private val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT)
    private var audioRecord: AudioRecord? = null
    private lateinit var statusTextView: TextView
    private lateinit var nightRecordingButton: Button
    private lateinit var listeningButton: Button
    private lateinit var waveBackground: WaveBackgroundView
    private var fftBuffer = FloatArray(BUFFER_SIZE / 2)
    private lateinit var amplitudeChartView: AmplitudeChartView



    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_detection, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        amplitudeChartView = requireActivity().findViewById(R.id.amplitudeChartView)
        statusTextView = view.findViewById(R.id.statusTextView)
        nightRecordingButton = view.findViewById(R.id.nightRecordingButton)
        listeningButton = view.findViewById(R.id.listeningButton)
        waveBackground = view.findViewById(R.id.waveBackground)

        nightRecordingButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
                amplitudeChartView.stopRecording()
                nightRecordingButton.text = "Запись ночью"
                statusTextView.text = "Запись остановлена"
            } else {
                if (ActivityCompat.checkSelfPermission(requireContext(), Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    startRecording(true)
                    amplitudeChartView.startRecording()
                    nightRecordingButton.text = "Остановить запись"
                    statusTextView.text = "Идёт запись ночью..."
                } else {
                    requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
                }
            }
        }

        listeningButton.setOnClickListener {
            if (isRecording) {
                stopRecording()
                amplitudeChartView.stopRecording()
                listeningButton.text = "Просто прослушивание"
                statusTextView.text = "Прослушивание остановлено"
            } else {
                if (ActivityCompat.checkSelfPermission(requireContext(), Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    startRecording(false)
                    amplitudeChartView.startRecording()
                    listeningButton.text = "Остановить прослушивание"
                    statusTextView.text = "Идёт прослушивание..."
                } else {
                    requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
                }
            }
        }

    }

    private fun startRecording(isNightMode: Boolean) {
        if (ActivityCompat.checkSelfPermission(requireContext(), Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            Log.e("AudioNN", "Permission to record audio not granted")
            activity?.runOnUiThread {
                statusTextView.text = "Ошибка: нет разрешения на запись"
            }
            return
        }

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE
            )
            audioRecord?.startRecording()
        } catch (e: SecurityException) {
            Log.e("AudioNN", "SecurityException: ${e.message}")
            activity?.runOnUiThread {
                statusTextView.text = "Ошибка доступа к микрофону"
            }
            return
        }
        isRecording = true
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            BUFFER_SIZE
        )

        audioRecord?.startRecording()
        CoroutineScope(Dispatchers.IO).launch {
            val audioBuffer = ShortArray(BUFFER_SIZE / 2)
            val directory = requireActivity().getExternalFilesDir(android.os.Environment.DIRECTORY_DOCUMENTS)
            if (directory == null) {
                Log.e("AudioNN", "Cannot access storage directory")
                activity?.runOnUiThread { statusTextView.text = "Ошибка: Нет доступа к хранилищу" }
                return@launch
            }

            val outputFile = File(directory, if (isNightMode) "night_output.txt" else "listening_output.txt")
            try {
                val outputStream = FileOutputStream(outputFile, true)
                while (isRecording) {
                    val read = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: 0
                    if (read > 0) {
                        val data = audioBuffer.joinToString(",")
                        outputStream.write("Raw audio data: $data\n".toByteArray())
                        Log.d("AudioNN", "Read $read samples: ${audioBuffer.take(10).joinToString()}")
                        updateVisualizer(audioBuffer)
                    }
                }
                audioRecord?.stop()
                audioRecord?.release()
                audioRecord = null
                outputStream.close()
            } catch (e: Exception) {
                Log.e("AudioNN", "Error writing to file: ${e.message}")
                activity?.runOnUiThread { statusTextView.text = "Ошибка: ${e.message}" }
            }
        }
    }

    private fun stopRecording() {
        isRecording = false
    }

    private fun updateVisualizer(audioBuffer: ShortArray) {
        for (i in fftBuffer.indices) {
            fftBuffer[i] = audioBuffer[i].toFloat() / Short.MAX_VALUE
        }
        val magnitudes = calculateMagnitude(fftBuffer)
        val averageAmplitude = magnitudes.average().toFloat() * 5000f
        waveBackground.updateAmplitude(averageAmplitude)
    }

    private fun calculateMagnitude(buffer: FloatArray): FloatArray {
        val result = FloatArray(buffer.size / 2)
        for (i in result.indices) {
            result[i] = kotlin.math.abs(buffer[i])
        }
        return result
    }

    override fun onDestroyView() {
        super.onDestroyView()
        stopRecording()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        if (requestCode == 1 && grantResults.all { it == PackageManager.PERMISSION_GRANTED }) {
            statusTextView.text = "Разрешение предоставлено, выберите режим"
        } else {
            Log.e("AudioNN", "Permission denied")
            statusTextView.text = "Ошибка: Разрешение на запись звука не предоставлено"
        }
    }
}