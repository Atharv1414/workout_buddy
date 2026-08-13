import 'package:test/test.dart';
import 'package:workout_buddy/models/exercise.dart';
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
}
