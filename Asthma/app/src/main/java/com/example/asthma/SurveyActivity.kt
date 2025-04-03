package com.example.asthma

import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class SurveyActivity : AppCompatActivity() {

    private lateinit var sharedPreferences: SharedPreferences

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_survey)

        // Инициализируем SharedPreferences
        sharedPreferences = getSharedPreferences("UserData", MODE_PRIVATE)

        val nameEditText: EditText = findViewById(R.id.nameEditText)
        val genderRadioGroup: RadioGroup = findViewById(R.id.genderRadioGroup)
        val maleRadioButton: RadioButton = findViewById(R.id.maleRadioButton)
        val femaleRadioButton: RadioButton = findViewById(R.id.femaleRadioButton)
        val weightEditText: EditText = findViewById(R.id.weightEditText)
        val heightEditText: EditText = findViewById(R.id.heightEditText)
        val submitButton: Button = findViewById(R.id.submitButton)

        // Если это редактирование, заполняем поля сохранёнными данными
        val savedName = sharedPreferences.getString("name", null)
        val savedGender = sharedPreferences.getString("gender", null)
        val savedWeight = sharedPreferences.getFloat("weight", -1f)
        val savedHeight = sharedPreferences.getFloat("height", -1f)

        if (savedName != null) {
            nameEditText.setText(savedName)
        }
        if (savedGender != null) {
            if (savedGender == "Мужской") {
                maleRadioButton.isChecked = true
            } else if (savedGender == "Женский") {
                femaleRadioButton.isChecked = true
            }
        }
        if (savedWeight > 0) {
            weightEditText.setText(savedWeight.toString())
        }
        if (savedHeight > 0) {
            heightEditText.setText(savedHeight.toString())
        }

        submitButton.setOnClickListener {
            // Получаем имя
            val name = nameEditText.text.toString().trim()
            if (name.isEmpty()) {
                Toast.makeText(this, "Пожалуйста, введите имя", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // Получаем выбранный пол
            val selectedGenderId = genderRadioGroup.checkedRadioButtonId
            if (selectedGenderId == -1) {
                Toast.makeText(this, "Пожалуйста, выберите пол", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val gender = if (selectedGenderId == maleRadioButton.id) "Мужской" else "Женский"

            // Получаем вес
            val weightText = weightEditText.text.toString()
            if (weightText.isEmpty()) {
                Toast.makeText(this, "Пожалуйста, введите вес", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val weight = weightText.toFloatOrNull()
            if (weight == null || weight <= 0) {
                Toast.makeText(this, "Пожалуйста, введите корректный вес", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // Получаем рост
            val heightText = heightEditText.text.toString()
            if (heightText.isEmpty()) {
                Toast.makeText(this, "Пожалуйста, введите рост", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val height = heightText.toFloatOrNull()
            if (height == null || height <= 0) {
                Toast.makeText(this, "Пожалуйста, введите корректный рост", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // Сохраняем данные в SharedPreferences
            with(sharedPreferences.edit()) {
                putString("name", name)
                putString("gender", gender)
                putFloat("weight", weight)
                putFloat("height", height)
                apply() // Асинхронно сохраняем данные
            }

            // Возвращаем данные в MainActivity
            val resultIntent = Intent().apply {
                putExtra("name", name)
                putExtra("gender", gender)
                putExtra("weight", weight)
                putExtra("height", height)
            }
            setResult(RESULT_OK, resultIntent)
            finish() // Закрываем SurveyActivity
        }
    }
}