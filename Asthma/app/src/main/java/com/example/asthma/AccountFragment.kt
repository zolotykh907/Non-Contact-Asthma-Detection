package com.example.asthma

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageButton
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.fragment.app.Fragment

class AccountFragment : Fragment() {

    private lateinit var userNameTextView: TextView
    private lateinit var userInfoTextView: TextView
    private lateinit var sharedPreferences: SharedPreferences

    private val editSurveyLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data = result.data
            val name = data?.getStringExtra("name") ?: "Не указано"
            val gender = data?.getStringExtra("gender") ?: "Не указано"
            val weight = data?.getFloatExtra("weight", -1f) ?: -1f
            val height = data?.getFloatExtra("height", -1f) ?: -1f

            // Отобразить данные на экране
            userNameTextView.text = name
            userInfoTextView.text = "Пол: $gender\nВес: $weight кг\nРост: $height см"

            // Сохранить данные и обновить историю веса
            val editor = sharedPreferences.edit()
            editor.putString("name", name)
            editor.putString("gender", gender)
            editor.putFloat("weight", weight)
            editor.putFloat("height", height)

            // Получаем старую историю веса
            val oldWeightsString = sharedPreferences.getString("weightHistory", "")
            val oldWeights = if (!oldWeightsString.isNullOrEmpty()) {
                oldWeightsString.split(",").mapNotNull { it.toFloatOrNull() }.toMutableList()
            } else {
                mutableListOf()
            }

            // Добавляем новый вес, если он валидный и не равен последнему
            if (weight > 0f && (oldWeights.isEmpty() || oldWeights.last() != weight)) {
                oldWeights.add(weight)
            }

            // Сохраняем обновлённую историю в строку
            editor.putString("weightHistory", oldWeights.joinToString(","))
            editor.apply()

            // Обновляем график
            val chartView = view?.findViewById<WeightChartView>(R.id.weightChartView)
            chartView?.weightData = oldWeights
            chartView?.invalidate()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        return inflater.inflate(R.layout.fragment_account, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        sharedPreferences = requireActivity().getSharedPreferences("UserData", Context.MODE_PRIVATE)

        userNameTextView = view.findViewById(R.id.userNameTextView)
        userInfoTextView = view.findViewById(R.id.userInfoTextView)
        val editButton: ImageButton = view.findViewById(R.id.editButton)

        val name = sharedPreferences.getString("name", "Не указано") ?: "Не указано"
        val gender = sharedPreferences.getString("gender", "Не указано") ?: "Не указано"
        val weight = sharedPreferences.getFloat("weight", -1f)
        val height = sharedPreferences.getFloat("height", -1f)

        userNameTextView.text = name
        userInfoTextView.text = "Пол: $gender\nВес: $weight кг\nРост: $height см"

        val savedWeightsString = sharedPreferences.getString("weightHistory", "")
        val savedWeights = if (!savedWeightsString.isNullOrEmpty()) {
            savedWeightsString.split(",").mapNotNull { it.toFloatOrNull() }
        } else {
            listOf()
        }

        val chartView = view.findViewById<WeightChartView>(R.id.weightChartView)
        chartView.weightData = savedWeights
        chartView.invalidate()

        editButton.setOnClickListener {
            val intent = Intent(requireContext(), SurveyActivity::class.java)
            editSurveyLauncher.launch(intent)
        }
    }
}
