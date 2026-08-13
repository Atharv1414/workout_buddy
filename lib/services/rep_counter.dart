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

  RepUpdate update(double angle) {
    if (exercise.type == ExerciseType.curl) {
      if (angle >= exercise.upAngle) stage = 'extended';
      if (angle <= exercise.downAngle) {
        if (stage == 'extended') reps++;
        stage = 'contracted';
      }
      return RepUpdate(
        reps,
        stage,
        stage == 'contracted' ? 'Lower with control' : 'Curl up',
      );
    }

    if (angle <= exercise.downAngle) stage = 'down';
    if (angle >= exercise.upAngle) {
      if (stage == 'down') reps++;
      stage = 'up';
    }
    return RepUpdate(
      reps,
      stage,
      stage == 'down' ? 'Drive up' : 'Lower with control',
    );
  }
}
