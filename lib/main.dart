import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/home_screen.dart';
import 'screens/onboarding_screen.dart';
import 'services/app_preferences.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Pose angles are tuned for an upright frame; the camera screen is portrait.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final onboardingComplete = await AppPreferences().onboardingComplete();
  runApp(WorkoutBuddyApp(onboardingComplete: onboardingComplete));
}

class WorkoutBuddyApp extends StatelessWidget {
  const WorkoutBuddyApp({super.key, required this.onboardingComplete});
  final bool onboardingComplete;

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF71E6A5);
    return MaterialApp(
      title: 'Workout Buddy',
      debugShowCheckedModeBanner: false,
      themeMode: ThemeMode.dark,
      darkTheme: ThemeData(
        brightness: Brightness.dark,
        colorScheme: ColorScheme.fromSeed(
          seedColor: seed,
          brightness: Brightness.dark,
        ),
        scaffoldBackgroundColor: const Color(0xFF0C1110),
        cardTheme: const CardThemeData(color: Color(0xFF18201D), elevation: 0),
        useMaterial3: true,
      ),
      home: onboardingComplete ? const HomeScreen() : const OnboardingScreen(),
    );
  }
}
