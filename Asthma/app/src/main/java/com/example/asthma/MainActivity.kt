package com.example.asthma

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.bottomnavigation.BottomNavigationView
import androidx.fragment.app.Fragment

class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val bottomNavigation: BottomNavigationView = findViewById(R.id.bottom_navigation)

        // Получаем данные из LauncherActivity
        val name = intent.getStringExtra("name") ?: "Не указано"
        val gender = intent.getStringExtra("gender") ?: "Не указано"
        val weight = intent.getFloatExtra("weight", -1f)
        val height = intent.getFloatExtra("height", -1f)

        // Устанавливаем начальный фрагмент (Детекция по умолчанию)
        if (savedInstanceState == null) {
            val detectionFragment = DetectionFragment()
            supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, detectionFragment)
                .commit()
            // Подсвечиваем вкладку "Детекция" в нижней навигации
            bottomNavigation.selectedItemId = R.id.nav_detection
        }

        // Переключение между вкладками
        bottomNavigation.setOnItemSelectedListener { item ->
            val fragment: Fragment = when (item.itemId) {
                R.id.nav_account -> {
                    val accountFragment = AccountFragment()
                    val bundle = Bundle().apply {
                        putString("name", name)
                        putString("gender", gender)
                        putFloat("weight", weight)
                        putFloat("height", height)
                    }
                    accountFragment.arguments = bundle
                    accountFragment
                }
                R.id.nav_detection -> DetectionFragment()
                R.id.nav_stats -> StatsFragment()
                else -> DetectionFragment()
            }
            supportFragmentManager.beginTransaction()
                .replace(R.id.fragment_container, fragment)
                .commit()
            true
        }
    }
}