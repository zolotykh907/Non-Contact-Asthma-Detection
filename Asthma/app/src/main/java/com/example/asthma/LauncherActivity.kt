package com.example.asthma

import android.content.Intent
import android.content.SharedPreferences
import android.os.Bundle
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class LauncherActivity : AppCompatActivity() {

    private lateinit var sharedPreferences: SharedPreferences

    // Лаунчер для получения результата из SurveyActivity
    private val surveyLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            // Получаем данные из SurveyActivity
            val data = result.data
            val name = data?.getStringExtra("name") ?: "Не указано"
            val gender = data?.getStringExtra("gender") ?: "Не указано"
            val weight = data?.getFloatExtra("weight", -1f) ?: -1f
            val height = data?.getFloatExtra("height", -1f) ?: -1f

            // Переходим в MainActivity
            val intent = Intent(this, MainActivity::class.java).apply {
                putExtra("name", name)
                putExtra("gender", gender)
                putExtra("weight", weight)
                putExtra("height", height)
            }
            startActivity(intent)
            finish() // Закрываем LauncherActivity
        } else {
            // Если пользователь закрыл SurveyActivity без сохранения, закрываем приложение
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Инициализируем SharedPreferences
        sharedPreferences = getSharedPreferences("UserData", MODE_PRIVATE)

        // Проверяем, есть ли сохранённые данные
        val name = sharedPreferences.getString("name", null)
        val gender = sharedPreferences.getString("gender", null)
        val weight = sharedPreferences.getFloat("weight", -1f)
        val height = sharedPreferences.getFloat("height", -1f)

        if (name != null && gender != null && weight > 0 && height > 0) {
            // Данные есть, переходим в MainActivity
            val intent = Intent(this, MainActivity::class.java).apply {
                putExtra("name", name)
                putExtra("gender", gender)
                putExtra("weight", weight)
                putExtra("height", height)
            }
            startActivity(intent)
            finish() // Закрываем LauncherActivity
        } else {
            // Данных нет, запускаем SurveyActivity и ждём результат
            val intent = Intent(this, SurveyActivity::class.java)
            surveyLauncher.launch(intent)
        }
    }
}