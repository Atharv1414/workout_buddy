package com.atharv1414.workoutbuddy

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Bundle

class PrivacyPolicyActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        startActivity(
            Intent(
                Intent.ACTION_VIEW,
                Uri.parse("https://github.com/Atharv1414/workout_buddy/blob/main/PRIVACY_POLICY.md"),
            ),
        )
        finish()
    }
}
