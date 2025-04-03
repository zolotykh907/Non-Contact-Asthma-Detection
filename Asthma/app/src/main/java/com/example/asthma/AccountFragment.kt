package com.example.asthma

import android.app.Activity // Добавлено
import android.content.Context // Добавлено
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
        if (result.resultCode ==  Activity.RESULT_OK) {
            val data = result.data
            val name = data?.getStringExtra("name") ?: "Не указано"
            val gender = data?.getStringExtra("gender") ?: "Не указано"
            val weight = data?.getFloatExtra("weight", -1f) ?: -1f
            val height = data?.getFloatExtra("height", -1f) ?: -1f

            userNameTextView.text = name
            userInfoTextView.text = "Пол: $gender\nВес: $weight кг\nРост: $height см"
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

        val name = arguments?.getString("name") ?: sharedPreferences.getString("name", "Не указано") ?: "Не указано"
        val gender = arguments?.getString("gender") ?: sharedPreferences.getString("gender", "Не указано") ?: "Не указано"
        val weight = arguments?.getFloat("weight") ?: sharedPreferences.getFloat("weight", -1f)
        val height = arguments?.getFloat("height") ?: sharedPreferences.getFloat("height", -1f)

        userNameTextView.text = name
        userInfoTextView.text = "Пол: $gender\nВес: $weight кг\nРост: $height см"

        editButton.setOnClickListener {
            val intent = Intent(requireContext(), SurveyActivity::class.java)
            editSurveyLauncher.launch(intent)
        }
    }
}