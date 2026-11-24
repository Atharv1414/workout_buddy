import 'dart:async';

import 'package:app_links/app_links.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'screens/activation_screen.dart';
import 'screens/home_screen.dart';
import 'screens/onboarding_screen.dart';
import 'services/app_preferences.dart';
import 'services/gym_license.dart';
import 'services/license_controller.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Pose angles are tuned for an upright frame; the camera screen is portrait.
  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);
  final onboardingComplete = await AppPreferences().onboardingComplete();
  final licenses = LicenseController();
  await licenses.load();
  runApp(
    WorkoutBuddyApp(onboardingComplete: onboardingComplete, licenses: licenses),
  );
}

class WorkoutBuddyApp extends StatefulWidget {
  const WorkoutBuddyApp({
    super.key,
    required this.onboardingComplete,
    required this.licenses,
  });
  final bool onboardingComplete;
  final LicenseController licenses;

  @override
  State<WorkoutBuddyApp> createState() => _WorkoutBuddyAppState();
}

class _WorkoutBuddyAppState extends State<WorkoutBuddyApp> {
  static const _defaultSeed = Color(0xFF71E6A5);
  final _messenger = GlobalKey<ScaffoldMessengerState>();
  late bool _onboarded = widget.onboardingComplete;
  StreamSubscription<Uri>? _links;

  @override
  void initState() {
    super.initState();
    // Gyms open workoutbuddy://activate?code=… from their own app or site.
    // The stream also delivers the link that launched the app.
    _links = AppLinks().uriLinkStream.listen(_onLink, onError: (_) {});
  }

  @override
  void dispose() {
    _links?.cancel();
    super.dispose();
  }

  Future<void> _onLink(Uri uri) async {
    if (uri.scheme != 'workoutbuddy' || uri.host != 'activate') return;
    final code = uri.queryParameters['code'];
    if (code == null) return;
    String message;
    try {
      final license = await widget.licenses.activate(code);
      message = 'Activated by ${license.gymName}';
    } on LicenseException catch (error) {
      message = error.message;
    }
    _messenger.currentState?.showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) => LicenseScope(
    controller: widget.licenses,
    child: ListenableBuilder(
      listenable: widget.licenses,
      builder: (context, _) {
        final gym = widget.licenses.license;
        final seed = gym == null ? _defaultSeed : Color(0xFF000000 | gym.color);
        return MaterialApp(
          title: 'Workout Buddy',
          debugShowCheckedModeBanner: false,
          scaffoldMessengerKey: _messenger,
          themeMode: ThemeMode.dark,
          darkTheme: ThemeData(
            brightness: Brightness.dark,
            colorScheme: ColorScheme.fromSeed(
              seedColor: seed,
              brightness: Brightness.dark,
            ),
            scaffoldBackgroundColor: const Color(0xFF0C1110),
            cardTheme: const CardThemeData(
              color: Color(0xFF18201D),
              elevation: 0,
            ),
            useMaterial3: true,
          ),
          home: !_onboarded
              ? OnboardingScreen(
                  onDone: () => setState(() => _onboarded = true),
                )
              : widget.licenses.unlocked
              ? const HomeScreen()
              : const ActivationScreen(),
        );
      },
    ),
  );
}
