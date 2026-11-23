import 'package:test/test.dart';
import 'package:workout_buddy/models/exercise.dart';
import 'package:workout_buddy/services/frame_rotation.dart';
import 'package:workout_buddy/services/rep_counter.dart';

void main() {
  test('calculates a right angle', () {
    expect(
      jointAngle(
        const JointPoint(1, 0),
        const JointPoint(0, 0),
        const JointPoint(0, 1),
      ),
      closeTo(90, .001),
    );
  });

  test('squat requires a full depth and extension cycle', () {
    final counter = RepCounter(Exercise.all.first);
    for (final angle in [170.0, 130.0, 90.0, 92.0, 130.0, 165.0]) {
      counter.update(angle);
    }
    expect(counter.reps, 1);
  });

  test('partial push-up is not counted', () {
    final counter = RepCounter(Exercise.all[1]);
    for (final angle in [165.0, 130.0, 120.0, 160.0]) {
      counter.update(angle);
    }
    expect(counter.reps, 0);
  });

  test('curl counts at full contraction', () {
    final counter = RepCounter(Exercise.all[2]);
    for (final angle in [160.0, 100.0, 50.0, 48.0]) {
      counter.update(angle);
    }
    expect(counter.reps, 1);
  });

  group('first rep', () {
    test('squat starting at the bottom does not count a free rep', () {
      final counter = RepCounter(Exercise.all.first);
      for (final angle in [90.0, 165.0]) {
        counter.update(angle);
      }
      expect(counter.reps, 0);
      for (final angle in [90.0, 165.0]) {
        counter.update(angle);
      }
      expect(counter.reps, 1);
    });

    test('curl starting contracted does not count a free rep', () {
      final counter = RepCounter(Exercise.all[2]);
      for (final angle in [50.0, 160.0]) {
        counter.update(angle);
      }
      expect(counter.reps, 0);
      counter.update(50);
      expect(counter.reps, 1);
    });

    test('asks for the start position before the first rep', () {
      final update = RepCounter(Exercise.all.first).update(120);
      expect(update.stage, 'ready');
      expect(update.feedback, 'Stand tall to start');
    });
  });

  test('restartCycle keeps reps but drops a half-finished rep', () {
    final counter = RepCounter(Exercise.all[1]);
    for (final angle in [160.0, 85.0, 160.0, 85.0]) {
      counter.update(angle);
    }
    expect(counter.reps, 1);
    counter.restartCycle();
    counter.update(160);
    expect(counter.reps, 1);
  });

  group('frameRotationDegrees', () {
    test('iOS uses the sensor orientation as is', () {
      expect(
        frameRotationDegrees(
          isAndroid: false,
          sensorOrientation: 90,
          frontCamera: true,
          deviceRotation: 270,
        ),
        90,
      );
    });

    test('Android front camera adds the device rotation', () {
      expect(
        frameRotationDegrees(
          isAndroid: true,
          sensorOrientation: 270,
          frontCamera: true,
          deviceRotation: 90,
        ),
        0,
      );
    });

    test('Android back camera subtracts the device rotation', () {
      expect(
        frameRotationDegrees(
          isAndroid: true,
          sensorOrientation: 90,
          frontCamera: false,
          deviceRotation: 270,
        ),
        180,
      );
    });

    test('unknown device rotation skips the frame', () {
      expect(
        frameRotationDegrees(
          isAndroid: true,
          sensorOrientation: 90,
          frontCamera: false,
          deviceRotation: null,
        ),
        isNull,
      );
    });
  });

  group('WorkoutRecord', () {
    test('round-trips through storage encoding', () {
      final record = WorkoutRecord(
        exercise: 'Squats',
        reps: 12,
        seconds: 95,
        completedAt: DateTime.utc(2026, 11, 20, 9, 30),
      );
      final decoded = WorkoutRecord.decode(record.encode())!;
      expect(decoded.exercise, 'Squats');
      expect(decoded.reps, 12);
      expect(decoded.seconds, 95);
      expect(decoded.completedAt, record.completedAt);
    });

    test('round-trips form data', () {
      final record = WorkoutRecord(
        exercise: 'Push-ups',
        reps: 10,
        seconds: 60,
        completedAt: DateTime.utc(2026, 11, 20),
        goodReps: 7,
        attempts: 11,
      );
      final decoded = WorkoutRecord.decode(record.encode())!;
      expect(decoded.goodReps, 7);
      expect(decoded.attempts, 11);
      expect(decoded.formScore, 64);
    });

    test('reads workouts saved before form scoring', () {
      final decoded = WorkoutRecord.decode(
        'Squats|12|95|2026-08-13T17:00:00.000',
      )!;
      expect(decoded.reps, 12);
      expect(decoded.goodReps, isNull);
      expect(decoded.formScore, isNull);
    });

    test('ignores corrupt entries', () {
      expect(WorkoutRecord.decode('Squats|x|95|2026-11-20'), isNull);
      expect(WorkoutRecord.decode('garbage'), isNull);
      expect(
        WorkoutRecord.decode('Squats|12|95|2026-08-13T17:00:00.000|x|11'),
        isNull,
      );
    });
  });
}
