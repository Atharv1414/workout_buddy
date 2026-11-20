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
      subtitle: 'Track knee depth and complete reps',
      emoji: '🏋️',
      setupTip:
          'Stand side-on to the phone, 2–3 m away, with hip, knee and ankle in view.',
      downAngle: 95,
      upAngle: 160,
    ),
    Exercise(
      type: ExerciseType.pushup,
      name: 'Push-ups',
      subtitle: 'Track elbow depth and lockout',
      emoji: '💪',
      setupTip:
          'Put the phone on the floor side-on, about 2 m away, with shoulder, elbow and wrist in view.',
      downAngle: 90,
      upAngle: 155,
    ),
    Exercise(
      type: ExerciseType.curl,
      name: 'Bicep curls',
      subtitle: 'Track contraction and extension',
      emoji: '🔥',
      setupTip:
          'Stand side-on, about 2 m away, with your working arm nearest the camera.',
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
