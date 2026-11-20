import 'dart:math' as math;

import '../models/exercise.dart';

class JointPoint {
  const JointPoint(this.x, this.y);
  final double x;
  final double y;
}

double jointAngle(JointPoint a, JointPoint b, JointPoint c) {
  final first = math.atan2(c.y - b.y, c.x - b.x);
  final second = math.atan2(a.y - b.y, a.x - b.x);
  var angle = (first - second).abs() * 180 / math.pi;
  if (angle > 180) angle = 360 - angle;
  return angle;
}

class RepUpdate {
  const RepUpdate(this.reps, this.stage, this.feedback);
  final int reps;
  final String stage;
  final String feedback;
}

class RepCounter {
  RepCounter(this.exercise);

  final Exercise exercise;
  int reps = 0;
  String stage = 'ready';

  /// Forgets the current position but keeps the rep count, for example when
  /// the tracked body side changes mid-set. The next rep must start from the
  /// fully extended position again.
  void restartCycle() => stage = 'ready';

  RepUpdate update(double angle) {
    if (exercise.type == ExerciseType.curl) {
      if (angle >= exercise.upAngle) stage = 'extended';
      if (angle <= exercise.downAngle && stage != 'ready') {
        if (stage == 'extended') reps++;
        stage = 'contracted';
      }
      return RepUpdate(reps, stage, switch (stage) {
        'ready' => 'Start with your arm straight',
        'contracted' => 'Lower with control',
        _ => 'Curl up',
      });
    }

    // A rep is extended -> down -> extended. Starting at the bottom does not
    // count, so the first rep needs the extended position first.
    if (angle >= exercise.upAngle) {
      if (stage == 'down') reps++;
      stage = 'up';
    } else if (angle <= exercise.downAngle && stage != 'ready') {
      stage = 'down';
    }
    return RepUpdate(reps, stage, switch (stage) {
      'ready' =>
        exercise.type == ExerciseType.squat
            ? 'Stand tall to start'
            : 'Start at the top',
      'down' => 'Drive up',
      _ => 'Lower with control',
    });
  }
}
