enum ExerciseType { squat, pushup, curl }

class Exercise {
  const Exercise({
    required this.type,
    required this.name,
    required this.subtitle,
    required this.emoji,
    required this.downAngle,
    required this.upAngle,
  });

  final ExerciseType type;
  final String name;
  final String subtitle;
  final String emoji;
  final double downAngle;
  final double upAngle;

  static const all = <Exercise>[
    Exercise(
      type: ExerciseType.squat,
      name: 'Squats',
      subtitle: 'Track knee depth and complete reps',
      emoji: '🏋️',
      downAngle: 95,
      upAngle: 160,
    ),
    Exercise(
      type: ExerciseType.pushup,
      name: 'Push-ups',
      subtitle: 'Track elbow depth and lockout',
      emoji: '💪',
      downAngle: 90,
      upAngle: 155,
    ),
    Exercise(
      type: ExerciseType.curl,
      name: 'Bicep curls',
      subtitle: 'Track contraction and extension',
      emoji: '🔥',
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
  });

  final String exercise;
  final int reps;
  final int seconds;
  final DateTime completedAt;

  String encode() =>
      '$exercise|$reps|$seconds|${completedAt.toIso8601String()}';

  static WorkoutRecord? decode(String value) {
    final parts = value.split('|');
    if (parts.length != 4) return null;
    final reps = int.tryParse(parts[1]);
    final seconds = int.tryParse(parts[2]);
    final date = DateTime.tryParse(parts[3]);
    if (reps == null || seconds == null || date == null) return null;
    return WorkoutRecord(
      exercise: parts[0],
      reps: reps,
      seconds: seconds,
      completedAt: date,
    );
  }
}
