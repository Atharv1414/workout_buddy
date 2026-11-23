import 'dart:math' as math;

import '../models/exercise.dart';
import 'rep_counter.dart';

/// Body joints on the tracked side. Image coordinates: x right, y down.
enum Joint { shoulder, elbow, wrist, hip, knee, ankle }

typedef PoseSample = Map<Joint, JointPoint>;

/// A form fault the coach can detect from a side view.
enum FormIssue {
  shallow(
    'Partial range',
    'Use the full range of motion. Partial reps are not counted.',
  ),
  torsoLean('Forward lean', 'Keep your chest up and sit back into your hips.'),
  hipSag(
    'Hips sagging',
    'Brace your core and squeeze your glutes to keep a straight line.',
  ),
  hipPike(
    'Hips too high',
    'Lower your hips until shoulders, hips and ankles line up.',
  ),
  elbowDrift(
    'Elbow drifting',
    'Keep your elbow pinned to your side so the biceps does the work.',
  ),
  bodySwing(
    'Body swing',
    'Stand tall and still. Lower the weight if you need momentum.',
  );

  const FormIssue(this.label, this.tip);

  /// Short name for summaries.
  final String label;

  /// Coaching advice for the end-of-set summary.
  final String tip;

  /// What to say during the set when this fault is detected.
  String cueFor(ExerciseType type) => switch (this) {
    shallow => switch (type) {
      ExerciseType.squat => 'Go deeper. That one didn\'t count',
      ExerciseType.pushup => 'Lower your chest further',
      ExerciseType.curl => 'Curl all the way up',
    },
    torsoLean => 'Chest up',
    hipSag => 'Lift your hips',
    hipPike => 'Lower your hips',
    elbowDrift => 'Keep your elbow tucked',
    bodySwing => 'Don\'t swing your body',
  };
}

/// The outcome of one rep attempt.
class RepResult {
  const RepResult({required this.counted, required this.issues});

  /// False for a partial attempt that the counter rejected.
  final bool counted;
  final List<FormIssue> issues;

  bool get clean => counted && issues.isEmpty;
}

class CoachUpdate {
  const CoachUpdate({
    required this.angle,
    required this.cue,
    required this.tone,
    this.rep,
  });

  final double angle;
  final String cue;
  final CueTone tone;

  /// Set on the frame a rep attempt finished.
  final RepResult? rep;
}

enum CueTone { neutral, good, warning }

/// Counts reps and checks form for every rep from side-view pose samples.
///
/// A rep cycle starts at the extended position, flexes past the exercise's
/// depth threshold and returns. Faults must hold for [persistFrames]
/// consecutive frames, so single noisy frames are ignored.
class FormCoach {
  FormCoach(this.exercise) : _counter = RepCounter(exercise);

  final Exercise exercise;
  final RepCounter _counter;
  final List<RepResult> results = [];

  static const persistFrames = 3;

  /// How far below the extended angle a movement must go to be treated as
  /// a rep attempt rather than fidgeting.
  static const attemptMargin = 25.0;

  /// How many frames a post-rep message stays on screen (~1 s at 15-20 fps).
  static const holdFrames = 18;

  // Thresholds, in degrees. Starting points; calibrate on real devices.
  static const maxSquatLean = 50.0;
  static const minPushupBodyLine = 160.0;
  static const maxCurlElbowDrift = 35.0;
  static const maxCurlSwing = 15.0;

  bool _cycleActive = false;
  bool _countedThisCycle = false;
  double _cycleMin = double.infinity;
  final Map<FormIssue, int> _streak = {};
  final Set<FormIssue> _cycleIssues = {};
  double? _leanMin;
  double? _leanMax;
  String? _heldCue;
  CueTone _heldTone = CueTone.neutral;
  int _holdLeft = 0;

  int get reps => _counter.reps;
  int get goodReps => results.where((r) => r.clean).length;
  int get attempts => results.length;

  /// Percentage of attempts that were counted with clean form, or null
  /// before the first attempt.
  int? get formScore =>
      attempts == 0 ? null : (goodReps * 100 / attempts).round();

  /// The most frequent fault in this set, if any.
  FormIssue? get topIssue {
    final counts = <FormIssue, int>{};
    for (final result in results) {
      for (final issue in result.issues) {
        counts[issue] = (counts[issue] ?? 0) + 1;
      }
    }
    if (counts.isEmpty) return null;
    return counts.entries.reduce((a, b) => b.value > a.value ? b : a).key;
  }

  /// Joints that must be visible to count reps at all.
  static List<Joint> requiredJoints(ExerciseType type) =>
      type == ExerciseType.squat
      ? const [Joint.hip, Joint.knee, Joint.ankle]
      : const [Joint.shoulder, Joint.elbow, Joint.wrist];

  /// Forgets the rep in progress, for example after switching body side.
  void restartCycle() {
    _counter.restartCycle();
    _cycleActive = false;
    _resetCycle(double.infinity);
  }

  /// Processes one frame. Returns null when a required joint is missing.
  CoachUpdate? update(PoseSample pose) {
    final joints = requiredJoints(exercise.type);
    if (joints.any((joint) => !pose.containsKey(joint))) return null;
    final angle = jointAngle(
      pose[joints[0]]!,
      pose[joints[1]]!,
      pose[joints[2]]!,
    );

    final repsBefore = _counter.reps;
    final phase = _counter.update(angle);
    _cycleMin = math.min(_cycleMin, angle);

    final frameIssues = _frameIssues(pose);
    for (final issue in FormIssue.values) {
      if (frameIssues.contains(issue)) {
        final streak = (_streak[issue] ?? 0) + 1;
        _streak[issue] = streak;
        if (streak >= persistFrames && _cycleActive) _cycleIssues.add(issue);
      } else {
        _streak.remove(issue);
      }
    }
    if (_cycleActive) _cycleIssues.addAll(_cycleRangeIssues());

    RepResult? finished;
    if (_counter.reps > repsBefore) {
      finished = RepResult(counted: true, issues: _ordered(_cycleIssues));
      _countedThisCycle = true;
      _hold(
        finished.clean
            ? 'Good rep'
            : finished.issues.first.cueFor(exercise.type),
        finished.clean ? CueTone.good : CueTone.warning,
      );
    }

    if (angle >= exercise.upAngle) {
      if (_cycleActive &&
          !_countedThisCycle &&
          _cycleMin < exercise.upAngle - attemptMargin) {
        finished = const RepResult(counted: false, issues: [FormIssue.shallow]);
        _hold(FormIssue.shallow.cueFor(exercise.type), CueTone.warning);
      }
      _cycleActive = true;
      _resetCycle(angle);
    }
    if (finished != null) results.add(finished);

    // A fault happening right now beats everything else.
    final live = _ordered(
      frameIssues.where((issue) => (_streak[issue] ?? 0) >= persistFrames),
    );
    if (live.isNotEmpty && _cycleActive) {
      return CoachUpdate(
        angle: angle,
        cue: live.first.cueFor(exercise.type),
        tone: CueTone.warning,
        rep: finished,
      );
    }
    if (_holdLeft > 0) {
      _holdLeft--;
      return CoachUpdate(
        angle: angle,
        cue: _heldCue!,
        tone: _heldTone,
        rep: finished,
      );
    }
    return CoachUpdate(
      angle: angle,
      cue: phase.feedback,
      tone: CueTone.neutral,
      rep: finished,
    );
  }

  void _hold(String cue, CueTone tone) {
    _heldCue = cue;
    _heldTone = tone;
    _holdLeft = holdFrames;
  }

  void _resetCycle(double angle) {
    _countedThisCycle = false;
    _cycleMin = angle;
    _cycleIssues.clear();
    _leanMin = null;
    _leanMax = null;
  }

  List<FormIssue> _ordered(Iterable<FormIssue> issues) =>
      issues.toSet().toList()..sort((a, b) => a.index.compareTo(b.index));

  /// Faults visible in a single frame.
  Set<FormIssue> _frameIssues(PoseSample pose) {
    final shoulder = pose[Joint.shoulder];
    final hip = pose[Joint.hip];
    switch (exercise.type) {
      case ExerciseType.squat:
        if (shoulder != null && hip != null) {
          if (torsoLean(hip, shoulder) > maxSquatLean) {
            return {FormIssue.torsoLean};
          }
        }
      case ExerciseType.pushup:
        final ankle = pose[Joint.ankle];
        if (shoulder != null && hip != null && ankle != null) {
          if (jointAngle(shoulder, hip, ankle) < minPushupBodyLine) {
            return {
              hipBelowLine(shoulder, hip, ankle)
                  ? FormIssue.hipSag
                  : FormIssue.hipPike,
            };
          }
        }
      case ExerciseType.curl:
        if (shoulder != null && hip != null) {
          final lean = torsoLean(hip, shoulder);
          _leanMin = math.min(_leanMin ?? lean, lean);
          _leanMax = math.max(_leanMax ?? lean, lean);
          final elbow = pose[Joint.elbow]!;
          if (jointAngle(hip, shoulder, elbow) > maxCurlElbowDrift) {
            return {FormIssue.elbowDrift};
          }
        }
    }
    return const {};
  }

  /// Faults that only show over the whole cycle, such as swinging.
  Set<FormIssue> _cycleRangeIssues() {
    final low = _leanMin;
    final high = _leanMax;
    if (exercise.type == ExerciseType.curl &&
        low != null &&
        high != null &&
        high - low > maxCurlSwing) {
      return {FormIssue.bodySwing};
    }
    return const {};
  }
}

/// Angle of the hip-to-shoulder line away from vertical, in degrees (0-180).
double torsoLean(JointPoint hip, JointPoint shoulder) {
  final dx = shoulder.x - hip.x;
  final dy = shoulder.y - hip.y;
  return math.atan2(dx.abs(), -dy) * 180 / math.pi;
}

/// Whether the hip sits below the straight shoulder-to-ankle line.
/// Image y grows downward, so "below" means a larger y.
bool hipBelowLine(JointPoint shoulder, JointPoint hip, JointPoint ankle) {
  final span = ankle.x - shoulder.x;
  if (span.abs() < 1e-6) return hip.y > shoulder.y;
  final t = (hip.x - shoulder.x) / span;
  final lineY = shoulder.y + t * (ankle.y - shoulder.y);
  return hip.y > lineY;
}
