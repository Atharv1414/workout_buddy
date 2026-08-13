# Workout Buddy

Workout Buddy is a privacy-first mobile exercise coach for iPhone and Android. It uses the phone camera and on-device ML Kit pose detection to count repetitions and give immediate movement cues for squats, push-ups, and bicep curls.

## Mobile app

- Native iOS and Android Flutter projects
- First-run privacy and safety onboarding
- Live front-camera preview and on-device pose estimation
- Stable, full-cycle repetition counting
- Left- and right-side tracking
- Squat, push-up, and bicep-curl coaching
- Workout timer, summary, and locally stored history
- Progress totals and recent sessions
- Remembered body-side preference and dedicated connection settings
- Camera retry, app background recovery, and discard protection
- No video upload or cloud requirement
- Optional, write-only workout export to Apple Health and Health Connect
- A fail-closed Technogym connection point ready for approved partner credentials

## Run the app

Install a current Flutter SDK, Xcode for iOS, and Android Studio for Android. Then:

```bash
flutter pub get
flutter run
```

Use a physical phone for camera and pose testing. The iOS target is 15.5 or newer and the Android minimum is API 26.

Useful checks:

```bash
flutter analyze
dart test
flutter build appbundle --release
flutter build ios --no-codesign
```

Android release builds require a private upload key configured from
`android/key.properties.example`. iOS archives require an Apple Developer Team
and signing profile in Xcode. Never commit those credentials.

See `RELEASE_CHECKLIST.md` for the signing, physical-device testing, store, and
partner-access steps that require the app-owner accounts.

## Architecture

- `lib/screens` contains the training, camera, and progress experiences.
- `lib/services/rep_counter.dart` contains testable angle and rep-counting logic.
- `lib/services/history_repository.dart` stores the latest 100 summaries locally.
- `lib/services/integrations.dart` provides Apple Health and Health Connect export and the safe Technogym boundary.
- `android` and `ios` contain their standard platform runners and camera permissions.

Pose processing runs on-device. A frame is skipped when its image format is unsupported, processing is already in progress, or the required joints are not visible enough. This keeps the live stream responsive and prevents low-confidence reps.

## Technogym roadmap

Technogym integration requires approval and credentials from its Mywellness/Enterprise integration team. Do not place an API key, authentication domain, or signing key in this repository or the mobile app.

The production flow should be:

1. The user signs in to Workout Buddy.
2. The app asks the Workout Buddy backend to connect Technogym.
3. The backend authenticates with Technogym using server-held credentials.
4. The backend returns only a short-lived, user-scoped token to the app.
5. Workouts are synchronized with idempotency identifiers and explicit user consent.

Until partner access is approved, Apple HealthKit and Android Health Connect are the interoperability layer. Health export is optional, requests write-only workout access, and keeps local saving independent of sync. The Technogym adapter deliberately fails closed instead of pretending a workout was synchronized.

## Legacy desktop prototype

The original Python prototype remains in `buddy.py`, with its dependencies in `requirements.txt`. It is independent from the Flutter mobile application.

> Workout Buddy is a fitness aid, not medical guidance. Stop exercising if you feel pain and consult a qualified professional when needed.
