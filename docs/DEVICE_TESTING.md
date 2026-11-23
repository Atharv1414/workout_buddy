# Device testing plan

Workout Buddy's camera, pose detection and form scoring have only been tested
with synthetic poses. ML Kit does not run in the iOS Simulator on Apple-silicon
Macs, so everything below needs a real phone. Work through the phases in order:
there is no point tuning thresholds (phase 3) until the basics (phase 2) work.

**Goal:** a live demo you can trust in front of a gym: reps counted correctly,
clean reps rarely flagged, and real faults caught.

---

## Phase 1 — Get the app on your phone (about 30 min)

### Before you start

- **Move the project out of `~/Documents`.** iCloud Drive syncs that folder
  and adds file metadata that makes iOS code signing fail ("resource fork,
  Finder information, or similar detritus not allowed"). It also creates the
  `" 2"` duplicate files. Use a local folder such as `~/Developer/workout_buddy`:

  ```bash
  mkdir -p ~/Developer && git clone https://github.com/Atharv1414/workout_buddy ~/Developer/workout_buddy
  ```

- Charge the phone and clear a 3 × 3 m floor space with good, even light.

### iPhone

1. Connect the iPhone by USB and unlock it. Tap **Trust** if asked.
2. On the phone: **Settings → Privacy & Security → Developer Mode → On**,
   then restart it.
3. Open `ios/Runner.xcworkspace` in Xcode (the workspace, not the
   `.xcodeproj`).
4. Select the **Runner** target → **Signing & Capabilities** → choose your
   Team (a free Apple ID works for testing). If Xcode says the bundle ID is
   taken, change it to something unique, such as
   `com.<yourname>.workoutbuddy.dev`, and don't commit that change.
5. If Xcode says the **HealthKit** capability isn't available for your team,
   remove HealthKit from Signing & Capabilities for this test build only.
   Health sync won't work, but everything else will. Don't commit that change.
6. From the project folder, run a release build (debug builds can't be
   reopened from the home screen):

   ```bash
   flutter run --release
   ```

7. First launch only, on the phone: **Settings → General → VPN & Device
   Management** → trust your developer certificate.

### Android (optional)

Needs Android Studio and its SDK installed. Turn on **Developer options →
USB debugging** on the phone, connect it, then run `flutter run --release`.

---

## Phase 2 — Smoke test (about 20 min)

Tick each item. If something fails, write down what you did and what you saw.

| # | Check | Expected | ✓ |
|---|---|---|---|
| 1 | Fresh install, open app | Onboarding shows 3 pages, then Train | |
| 2 | Open Squats, **deny** camera | "Camera access is turned off…" with Try again | |
| 3 | Allow camera in iOS Settings, return, tap Try again | Live front-camera preview | |
| 4 | Stand in view | Setup tip at top; cue says "Stand tall to start" | |
| 5 | Do 3 slow, deep squats | Counter 1 → 2 → 3, "Good rep" flashes green | |
| 6 | Tap Left/Right mid-set | No extra rep counted | |
| 7 | Reset | Reps and timer go to 0 | |
| 8 | Swipe home for 10 s, come back | Preview returns, timer did not run while away | |
| 9 | Receive a call (or open Control Center) mid-set, then return | Preview returns, reps kept | |
| 10 | Hold the phone sideways (landscape) | App stays portrait and still tracks | |
| 11 | Finish workout | Summary: reps, good form, score, focus tip | |
| 12 | Progress tab | New workout with score; Avg form updated | |
| 13 | Close with reps unsaved | "Discard workout?" dialog | |
| 14 | Settings → Apple Health on, finish a workout | Workout appears once in the Health app | |
| 15 | Repeat 4–11 for Push-ups and Bicep curls | Same behaviour | |
| 16 | Dim light / dark clothing on dark background | Note when tracking drops ("Keep your … side visible") | |

---

## Phase 3 — Accuracy and calibration (1–2 hours)

### Who and where

- **3–5 people** with different heights and builds. At least one should be
  experienced (a trainer, ideally) and one a beginner.
- Phone **side-on at hip height** (squats and curls) or **on the floor**
  (push-ups), **2–3 m away**, whole body in view.
- Test both the **left** and the **right** side, at 2 m and at 3 m.

### Protocol for each person and exercise

Do these four sets. Someone else counts and notes what really happened.

| Set | What to do | What the app should do |
|---|---|---|
| A — Clean | 10 slow reps with good form | 10 reps, ~10 good, no fault tip |
| B — Partial | 5 half-range reps | 0 reps, 5 partial attempts, "Partial range" tip |
| C — Fault | 5 full reps with the deliberate fault below | 5 reps, most flagged with the right fault |
| D — Fast | 10 reps at normal gym pace | 10 reps |

Deliberate faults for set C:

| Exercise | Fault to perform | Expected tip |
|---|---|---|
| Squat | Let the chest drop well forward | Forward lean |
| Push-up | Let the hips sag toward the floor (then repeat with hips piked up) | Hips sagging / Hips too high |
| Curl | Let the elbow drift forward, then repeat swinging the body | Elbow drifting / Body swing |

### Record the results

Copy this table into a spreadsheet, one row per set:

| Person | Exercise | Side | Distance | Set | Real reps | App reps | App good reps | App top fault | Notes |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

### Targets before pitching

| Measure | How to work it out | Target |
|---|---|---|
| Rep count accuracy | Sets A and D: app reps = real reps | ±0 on ≥ 90% of sets |
| False faults | Set A: reps flagged but done cleanly | ≤ 1 in 10 |
| Fault detection | Set C: deliberate faults flagged | ≥ 4 in 5 |
| Partial-rep detection | Set B: partial reps not counted | 5 in 5 |

---

## Phase 4 — Tuning the thresholds

The limits live in two files. Change **one at a time**, rebuild
(`flutter run --release`), and repeat only the sets that failed.

| Symptom | What to change | File |
|---|---|---|
| Clean squats flagged "Forward lean" | Raise `maxSquatLean` (50) by 5 | `lib/services/form_coach.dart` |
| Leaning squats not flagged | Lower `maxSquatLean` by 5 | same |
| Clean push-ups flagged sagging/piked | Lower `minPushupBodyLine` (160) by 5 | same |
| Sagging push-ups not flagged | Raise `minPushupBodyLine` by 5 | same |
| Strict curls flagged "Elbow drifting" | Raise `maxCurlElbowDrift` (35) by 5 | same |
| Strict curls flagged "Body swing" | Raise `maxCurlSwing` (15) by 5 | same |
| Faults flagged from one twitchy moment | Raise `persistFrames` (3) to 4–5 | same |
| Deep reps not counted | Raise that exercise's `downAngle` by 5 | `lib/models/exercise.dart` |
| Shallow reps counted | Lower that exercise's `downAngle` by 5 | same |
| Reps missed at the top | Lower that exercise's `upAngle` by 5 | same |

The live angle under the cue ("95° • 42s") is the joint angle the counter
uses: knee for squats, elbow for push-ups and curls. Watching it during a
rep shows how deep people really go.

After any change, run the tests (`flutter test`). Some tests use the current
limits, so update their numbers if a change is intentional.

---

## Phase 5 — Trainer review and demo video

1. Have a qualified trainer watch a few sets and read every cue and tip in
   the app. Note any wording they'd change.
2. Record the demo: screen-record the phone (Control Center → Screen
   Recording) while someone does one clean set and one set with a fault,
   then show the summary and the Progress tab. Keep it under 60 seconds.

## Done when

- [ ] All of phase 2 passes on at least one iPhone
- [ ] Phase 3 targets met for all three exercises
- [ ] A trainer has reviewed the cues
- [ ] A demo video is recorded
