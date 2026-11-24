import 'package:test/test.dart';
import 'package:workout_buddy/models/exercise.dart';
import 'package:workout_buddy/services/cue_policy.dart';
import 'package:workout_buddy/services/form_coach.dart';

Duration ms(int value) => Duration(milliseconds: value);

CoachUpdate frame(String cue, CueTone tone, [RepResult? rep]) =>
    CoachUpdate(angle: 120, cue: cue, tone: tone, rep: rep);

void main() {
  late CuePolicy policy;
  final coach = FormCoach(
    Exercise.all.firstWhere((e) => e.type == ExerciseType.squat),
  );
  setUp(() => policy = CuePolicy());

  SpokenCue? at(int time, CoachUpdate update) =>
      policy.onUpdate(update, coach, ms(time));

  const clean = RepResult(counted: true, issues: []);
  const leaning = RepResult(counted: true, issues: [FormIssue.torsoLean]);
  const partial = RepResult(counted: false, issues: [FormIssue.shallow]);
  final lean = frame('Chest up', CueTone.warning);

  test('a clean rep is just its number and interrupts', () {
    final cue = at(0, frame('Good rep', CueTone.good, clean));
    expect(cue!.text, '${coach.reps}');
    expect(cue.interrupt, isTrue);
  });

  test('a faulty rep adds advice, unless it was just said', () {
    final rep = frame('Chest up', CueTone.warning, leaning);
    expect(at(0, rep)!.text, '${coach.reps}. Chest up');
    expect(at(2000, rep)!.text, '${coach.reps}');
    expect(at(4000, rep)!.text, '${coach.reps}. Chest up');
  });

  test('a partial rep gives advice without a number', () {
    final cue = at(0, frame('x', CueTone.warning, partial));
    expect(cue!.text, FormIssue.shallow.cueFor(ExerciseType.squat));
  });

  test('a live fault is said once per gap', () {
    expect(at(0, lean)!.text, 'Chest up');
    expect(at(100, lean), isNull);
    expect(at(3999, lean), isNull);
    expect(at(4000, lean)!.text, 'Chest up');
  });

  test('neutral and positive frames are silent', () {
    expect(at(0, frame('Drive up', CueTone.neutral)), isNull);
    expect(at(0, frame('Good rep', CueTone.good)), isNull);
  });

  test('a live fault waits until the count has been heard', () {
    at(0, frame('Good rep', CueTone.good, clean));
    expect(at(500, lean), isNull);
    expect(at(1200, lean)!.text, 'Chest up');
  });

  test('out of view is announced after a delay, then rarely', () {
    SpokenCue? lost(int time) => policy.onLost('Step back', ms(time));
    expect(lost(0), isNull);
    expect(lost(1999), isNull);
    expect(lost(2000)!.text, 'Step back');
    expect(lost(5000), isNull);
    expect(lost(10000), isNotNull);
  });

  test('being seen again restarts the out-of-view delay', () {
    policy.onLost('Step back', ms(0));
    at(1500, frame('Drive up', CueTone.neutral));
    expect(policy.onLost('Step back', ms(2500)), isNull);
  });
}
