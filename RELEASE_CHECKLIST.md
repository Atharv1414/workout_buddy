# Workout Buddy release checklist

The v1 source is prepared for release. Items marked complete are in the repository;
the remaining items require private developer credentials, physical devices, store
accounts, or Technogym approval.

## Completed in source

- [x] Production Flutter projects for iPhone and Android with the stable identifier `com.atharv1414.workoutbuddy`.
- [x] Branded app icons, adaptive Android icon, and launch screens.
- [x] Camera permission handling, lifecycle recovery, workout discard protection, and on-device pose processing.
- [x] Local workout history plus optional, write-only Apple Health / Health Connect workout export.
- [x] HealthKit entitlements and usage descriptions; Health Connect manifest permissions and rationale screen.
- [x] Android target API 36, minimum API 26, minification, resource shrinking, and release-key configuration.
- [x] Privacy policy, store copy, reviewer instructions, and continuous integration checks.
- [x] Static analysis and automated Dart/Python tests passing.
- [x] Android release App Bundle build verified locally. It intentionally remains unsigned until the private upload key is supplied.
- [x] iOS CocoaPods dependencies installed and the Xcode project/workspace validated.

## Private signing setup

- [ ] Create or locate the Google Play upload keystore. Copy `android/key.properties.example` to `android/key.properties`, use absolute or app-relative keystore paths, and fill in the four private values. Both files are ignored by Git.
- [ ] Run `flutter build appbundle --release`, verify that the resulting AAB is signed, and upload it to a Play internal-testing track.
- [ ] Open `ios/Runner.xcworkspace` in Xcode, select the app owner's Apple Developer Team, enable automatic signing or choose the intended profiles, and confirm the HealthKit capability.
- [ ] Set the final App Store version/build, archive with Xcode, validate the archive, and upload it to TestFlight.

## Physical-device acceptance

- [ ] Test squats, push-ups, and curls on representative iPhones and Android devices.
- [ ] Calibrate pose thresholds with different body types, camera distances, clothing, and lighting.
- [ ] Test camera denial and later approval, Health permission denial and approval, background/resume, incoming calls, and low-memory recovery.
- [ ] Confirm that completed workouts stay local when health sync is off or fails, and appear once in Apple Health / Health Connect when enabled.
- [ ] Review all movement and safety cues with a qualified fitness professional.

## Store submission

- [ ] Push `PRIVACY_POLICY.md` to the public `main` branch so the listing URL resolves before review.
- [ ] Create App Store Connect and Google Play Console records for the final identifiers.
- [ ] Add screenshots for required phone sizes, support contact details, category, age rating, and the copy in `store/STORE_LISTING.md`.
- [ ] Paste `store/REVIEW_NOTES.md` into the reviewer notes and provide any requested demo access.
- [ ] Complete Apple privacy labels and Google Play Data Safety / Health Apps declarations from the behavior documented in `PRIVACY_POLICY.md`.
- [ ] Complete TestFlight and Play closed testing, fix blocking feedback, then use a staged production rollout.

## Technogym follow-up

- [ ] Apply to Technogym/Mywellness with the company, product, website, contact, and integration purpose.
- [ ] Build a backend that keeps partner credentials off the phone and issues only short-lived, user-scoped tokens.
- [ ] Add idempotent workout synchronization, bounded retries, disconnect, and data-deletion flows.
- [ ] Complete Technogym certification before enabling the currently pending connection tile.
