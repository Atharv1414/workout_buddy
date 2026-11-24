import 'form_coach.dart';

/// Something to say aloud. [interrupt] cuts off speech in progress.
class SpokenCue {
  const SpokenCue(this.text, {this.interrupt = false});
  final String text;
  final bool interrupt;
}

/// Decides what to say and when, so the coach talks like a trainer instead of
/// reading out every frame. Times are the caller's monotonic clock.
class CuePolicy {
  /// The same fault is not repeated within this time.
  static const repeatGap = Duration(seconds: 4);

  /// Quiet time after a rep, so its count can be heard.
  static const repGap = Duration(milliseconds: 1200);

  /// How long the body must be out of view before saying so, and how often.
  static const lostDelay = Duration(seconds: 2);
  static const lostGap = Duration(seconds: 8);

  final Map<String, Duration> _saidAt = {};
  Duration? _lastRep;
  Duration? _lostSince;
  Duration? _lostSaid;

  /// Call for each analysed frame; [reps] is the total after it.
  SpokenCue? onUpdate(CoachUpdate update, FormCoach coach, Duration now) {
    _lostSince = null;
    final rep = update.rep;
    if (rep != null) {
      _lastRep = now;
      final issue = rep.issues.firstOrNull?.cueFor(coach.exercise.type);
      // Skip the fault if it was just called out during the rep.
      final advice = issue != null && _due(issue, now) ? issue : null;
      if (advice != null) _saidAt[advice] = now;
      final parts = [if (rep.counted) '${coach.reps}', ?advice];
      return parts.isEmpty
          ? null
          : SpokenCue(parts.join('. '), interrupt: true);
    }
    if (update.tone != CueTone.warning || !_due(update.cue, now)) return null;
    final lastRep = _lastRep;
    if (lastRep != null && now - lastRep < repGap) return null;
    _saidAt[update.cue] = now;
    return SpokenCue(update.cue);
  }

  /// Call when the body, or the tracked side, is not visible.
  SpokenCue? onLost(String message, Duration now) {
    final since = _lostSince ??= now;
    if (now - since < lostDelay) return null;
    final said = _lostSaid;
    if (said != null && now - said < lostGap) return null;
    _lostSaid = now;
    return SpokenCue(message);
  }

  bool _due(String cue, Duration now) {
    final said = _saidAt[cue];
    return said == null || now - said >= repeatGap;
  }
}
