import 'dart:math' as math;

import 'package:test/test.dart';
import 'package:workout_buddy/models/exercise.dart';
import 'package:workout_buddy/services/form_coach.dart';
import 'package:workout_buddy/services/rep_counter.dart';

final squat = Exercise.all[0];
final pushup = Exercise.all[1];
final curl = Exercise.all[2];

/// Point at [length] from [origin] in direction [degrees] (0 = right,
/// 90 = down, image coordinates).
JointPoint toward(JointPoint origin, double degrees, double length) {
  final radians = degrees * math.pi / 180;
  return JointPoint(
    origin.x + length * math.cos(radians),
    origin.y + length * math.sin(radians),
  );
}

/// Side-view squat: shin vertical, [knee] = hip-knee-ankle angle,
/// [lean] = torso angle from vertical, leaning forward (+x).
PoseSample squatPose(double knee, {double lean = 20}) {
  const ankle = JointPoint(0, 100);
  const kneePoint = JointPoint(0, 60);
  final hip = toward(kneePoint, 90 + knee, 40);
  final shoulder = toward(hip, 270 + lean, 50);
  return {
    Joint.ankle: ankle,
    Joint.knee: kneePoint,
    Joint.hip: hip,
    Joint.shoulder: shoulder,
  };
}

/// Side-view push-up: [elbow] = shoulder-elbow-wrist angle, [sag] moves the
/// hip down (positive) or up (negative) from the straight body line.
PoseSample pushupPose(double elbow, {double sag = 0}) {
  const shoulder = JointPoint(0, 50);
  final elbowPoint = toward(shoulder, 90, 25);
  final wrist = toward(elbowPoint, 270 + elbow, 25);
  return {
    Joint.shoulder: shoulder,
    Joint.elbow: elbowPoint,
    Joint.wrist: wrist,
    Joint.hip: JointPoint(60, 50 + sag),
    Joint.ankle: const JointPoint(120, 50),
  };
}

/// Side-view curl: [elbow] = shoulder-elbow-wrist angle, [drift] = upper arm
/// angle away from the torso, [lean] = torso angle from vertical.
PoseSample curlPose(double elbow, {double drift = 5, double lean = 0}) {
  const hip = JointPoint(0, 60);
  final shoulder = toward(hip, 270 + lean, 60);
  final elbowPoint = toward(shoulder, 90 - drift, 30);
  final wrist = toward(elbowPoint, 270 - drift - elbow, 25);
  return {
    Joint.hip: hip,
    Joint.shoulder: shoulder,
    Joint.elbow: elbowPoint,
    Joint.wrist: wrist,
  };
}

/// Feeds each pose for several frames, like a camera at ~15 fps would.
List<CoachUpdate?> play(
  FormCoach coach,
  Iterable<PoseSample> poses, {
  int frames = 4,
}) => [
  for (final pose in poses)
    for (var i = 0; i < frames; i++) coach.update(pose),
];

void main() {
  test('synthetic poses produce the intended joint angles', () {
    final s = squatPose(90, lean: 30);
    expect(
      jointAngle(s[Joint.hip]!, s[Joint.knee]!, s[Joint.ankle]!),
      closeTo(90, .01),
    );
    expect(torsoLean(s[Joint.hip]!, s[Joint.shoulder]!), closeTo(30, .01));
    final c = curlPose(60, drift: 20);
    expect(
      jointAngle(c[Joint.shoulder]!, c[Joint.elbow]!, c[Joint.wrist]!),
      closeTo(60, .01),
    );
    expect(
      jointAngle(c[Joint.hip]!, c[Joint.shoulder]!, c[Joint.elbow]!),
      closeTo(20, .01),
    );
  });

  group('squat', () {
    test('clean reps score 100%', () {
      final coach = FormCoach(squat);
      for (var i = 0; i < 3; i++) {
        play(coach, [squatPose(170), squatPose(120), squatPose(85)]);
      }
      play(coach, [squatPose(170)]);
      expect(coach.reps, 3);
      expect(coach.goodReps, 3);
      expect(coach.formScore, 100);
      expect(coach.topIssue, isNull);
    });

    test('shows "Good rep" after a clean rep', () {
      final coach = FormCoach(squat);
      final updates = play(coach, [
        squatPose(170),
        squatPose(85),
        squatPose(170),
      ]);
      final repFrame = updates.indexWhere((u) => u?.rep != null);
      expect(updates[repFrame]!.rep!.clean, isTrue);
      expect(updates[repFrame]!.cue, 'Good rep');
      expect(updates[repFrame]!.tone, CueTone.good);
    });

    test('forward lean is counted but flagged', () {
      final coach = FormCoach(squat);
      play(coach, [squatPose(170), squatPose(85, lean: 65), squatPose(170)]);
      expect(coach.reps, 1);
      expect(coach.goodReps, 0);
      expect(coach.results.single.issues, [FormIssue.torsoLean]);
      expect(coach.topIssue, FormIssue.torsoLean);
    });

    test('a shallow attempt is not counted and lowers the score', () {
      final coach = FormCoach(squat);
      play(coach, [squatPose(170), squatPose(85), squatPose(170)]);
      final updates = play(coach, [squatPose(120), squatPose(170)]);
      expect(coach.reps, 1);
      expect(coach.attempts, 2);
      expect(coach.formScore, 50);
      expect(coach.results.last.counted, isFalse);
      expect(updates.last!.cue, 'Go deeper. That one didn\'t count');
    });

    test('small movements at the top are not attempts', () {
      final coach = FormCoach(squat);
      play(coach, [squatPose(170), squatPose(150), squatPose(170)]);
      expect(coach.attempts, 0);
      expect(coach.formScore, isNull);
    });

    test('a single noisy frame is not a fault', () {
      final coach = FormCoach(squat);
      play(coach, [squatPose(170)]);
      play(coach, [squatPose(85, lean: 80)], frames: 1);
      play(coach, [squatPose(85), squatPose(170)]);
      expect(coach.goodReps, 1);
    });

    test('returns null when a required joint is missing', () {
      final pose = squatPose(170)..remove(Joint.knee);
      expect(FormCoach(squat).update(pose), isNull);
    });

    test('form checks are skipped when the shoulder is not visible', () {
      final coach = FormCoach(squat);
      PoseSample noShoulder(double knee) =>
          squatPose(knee, lean: 70)..remove(Joint.shoulder);
      play(coach, [noShoulder(170), noShoulder(85), noShoulder(170)]);
      expect(coach.goodReps, 1);
    });
  });

  group('push-up', () {
    test('sagging hips are flagged', () {
      final coach = FormCoach(pushup);
      play(coach, [pushupPose(165), pushupPose(80, sag: 20), pushupPose(165)]);
      expect(coach.reps, 1);
      expect(coach.results.single.issues, [FormIssue.hipSag]);
    });

    test('piked hips are flagged', () {
      final coach = FormCoach(pushup);
      play(coach, [pushupPose(165), pushupPose(80, sag: -20), pushupPose(165)]);
      expect(coach.results.single.issues, [FormIssue.hipPike]);
    });

    test('a straight body line is clean', () {
      final coach = FormCoach(pushup);
      play(coach, [pushupPose(165), pushupPose(80), pushupPose(165)]);
      expect(coach.formScore, 100);
    });

    test('live cue warns while the fault is happening', () {
      final coach = FormCoach(pushup);
      play(coach, [pushupPose(165)]);
      final updates = play(coach, [pushupPose(120, sag: 25)]);
      expect(updates.last!.cue, 'Lift your hips');
      expect(updates.last!.tone, CueTone.warning);
    });
  });

  group('curl', () {
    test('clean curl is counted at the top of the curl', () {
      final coach = FormCoach(curl);
      play(coach, [curlPose(160), curlPose(100), curlPose(45)]);
      expect(coach.reps, 1);
      expect(coach.formScore, 100);
    });

    test('elbow drift is flagged', () {
      final coach = FormCoach(curl);
      play(coach, [
        curlPose(160),
        curlPose(100, drift: 45),
        curlPose(45, drift: 50),
      ]);
      expect(coach.results.single.issues, [FormIssue.elbowDrift]);
    });

    test('swinging the body is flagged', () {
      final coach = FormCoach(curl);
      play(coach, [
        curlPose(160),
        curlPose(110, lean: 20),
        curlPose(45, lean: 0),
      ]);
      expect(coach.results.single.issues, contains(FormIssue.bodySwing));
    });

    test('a half curl is a partial attempt', () {
      final coach = FormCoach(curl);
      play(coach, [curlPose(160), curlPose(100), curlPose(160)]);
      expect(coach.reps, 0);
      expect(coach.results.single.issues, [FormIssue.shallow]);
    });
  });

  test('restartCycle drops the rep in progress but keeps results', () {
    final coach = FormCoach(squat);
    play(coach, [squatPose(170), squatPose(85), squatPose(170)]);
    play(coach, [squatPose(85)]);
    coach.restartCycle();
    play(coach, [squatPose(170)]);
    expect(coach.reps, 1);
    expect(coach.attempts, 1);
  });

  test('top issue is the most frequent fault', () {
    final coach = FormCoach(pushup);
    play(coach, [pushupPose(165)]);
    for (final sag in [20.0, 20.0, -20.0]) {
      play(coach, [pushupPose(80, sag: sag), pushupPose(165)]);
    }
    expect(coach.topIssue, FormIssue.hipSag);
    expect(coach.formScore, 0);
  });
}
