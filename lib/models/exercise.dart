enum ExerciseType { squat, pushup, curl }

class Exercise {
  const Exercise({
    required this.type,
    required this.name,
    required this.subtitle,
    required this.emoji,
    required this.setupTip,
    required this.downAngle,
    required this.upAngle,
  });

  final ExerciseType type;
  final String name;
  final String subtitle;
  final String emoji;

  /// Where to put the phone. Joint angles are only reliable from the side.
  final String setupTip;
  final double downAngle;
  final double upAngle;

  static const all = <Exercise>[
    Exercise(
      type: ExerciseType.squat,
      name: 'Squats',
      subtitle: 'Depth and chest position on every rep',
      emoji: '🏋️',
      setupTip:
          'Stand side-on to the phone, 2–3 m away, with your whole body from head to feet in view.',
      downAngle: 95,
      upAngle: 160,
    ),
    Exercise(
      type: ExerciseType.pushup,
      name: 'Push-ups',
      subtitle: 'Depth and a straight body line',
      emoji: '💪',
      setupTip:
          'Put the phone on the floor side-on, about 2 m away, with your whole body from shoulders to feet in view.',
      downAngle: 90,
      upAngle: 155,
    ),
    Exercise(
      type: ExerciseType.curl,
      name: 'Bicep curls',
      subtitle: 'Full range, tucked elbow, no swinging',
      emoji: '🔥',
      setupTip:
          'Stand side-on, about 2 m away, with your working arm nearest the camera and your hips in view.',
      downAngle: 55,
      upAngle: 150,
    ),
  ];
}

class WorkoutRecord {
  const WorkoutRecord({
    required this.exercise,
    required this.reps,
    required this.seconds,
    required this.completedAt,
    this.goodReps,
    this.attempts,
  });

  final String exercise;
  final int reps;
  final int seconds;
  final DateTime completedAt;

  /// Reps with clean form. Null for workouts saved before form scoring.
  final int? goodReps;

  /// Counted reps plus rejected partial attempts. Null before form scoring.
  final int? attempts;

  /// Percentage of attempts that were clean reps, when known.
  int? get formScore {
    final good = goodReps;
    final total = attempts;
    if (good == null || total == null || total == 0) return null;
    return (good * 100 / total).round();
  }

  String encode() => [
    exercise,
    reps,
    seconds,
    completedAt.toIso8601String(),
    if (goodReps != null && attempts != null) ...[goodReps, attempts],
  ].join('|');

  /// Reads both the original 4-field format and the 6-field format with
  /// form data.
  static WorkoutRecord? decode(String value) {
    final parts = value.split('|');
    if (parts.length != 4 && parts.length != 6) return null;
    final reps = int.tryParse(parts[1]);
    final seconds = int.tryParse(parts[2]);
    final date = DateTime.tryParse(parts[3]);
    if (reps == null || seconds == null || date == null) return null;
    int? goodReps;
    int? attempts;
    if (parts.length == 6) {
      goodReps = int.tryParse(parts[4]);
      attempts = int.tryParse(parts[5]);
      if (goodReps == null || attempts == null) return null;
    }
    return WorkoutRecord(
      exercise: parts[0],
      reps: reps,
      seconds: seconds,
      completedAt: date,
      goodReps: goodReps,
      attempts: attempts,
    );
  }
}
