import 'dart:async';
import 'dart:io';

import 'package:flutter_tts/flutter_tts.dart';

import 'cue_policy.dart';

/// Speaks cues with the phone's own voice. Speech is a bonus: any failure is
/// swallowed so it can never interrupt a workout.
class VoiceCoach {
  VoiceCoach() {
    _ready = _guard(_setUp);
  }

  final FlutterTts _tts = FlutterTts();
  late final Future<void> _ready;
  bool _speaking = false;
  bool _disposed = false;
  bool enabled = true;

  Future<void> _setUp() async {
    void done() => _speaking = false;
    _tts
      ..setCompletionHandler(done)
      ..setCancelHandler(done)
      ..setErrorHandler((_) => done());
    if (Platform.isIOS) {
      // Audible with the silent switch on; lowers music instead of stopping it.
      await _tts.setIosAudioCategory(IosTextToSpeechAudioCategory.playback, [
        IosTextToSpeechAudioCategoryOptions.mixWithOthers,
        IosTextToSpeechAudioCategoryOptions.duckOthers,
      ]);
    }
    await _tts.setLanguage('en-US');
  }

  void say(SpokenCue? cue) {
    if (cue == null || !enabled || _disposed) return;
    if (_speaking && !cue.interrupt) return;
    unawaited(_guard(() => _speak(cue)));
  }

  Future<void> _speak(SpokenCue cue) async {
    await _ready;
    if (_disposed || !enabled) return;
    if (_speaking) await _tts.stop();
    _speaking = true;
    // `focus` asks Android to duck music while speaking.
    final started = await _tts.speak(cue.text, focus: true);
    if (started != 1) _speaking = false;
  }

  Future<void> stop() => _guard(() async {
    _speaking = false;
    await _tts.stop();
  });

  Future<void> dispose() {
    _disposed = true;
    return stop();
  }

  Future<void> _guard(Future<void> Function() action) async {
    try {
      await action();
    } catch (_) {
      _speaking = false;
    }
  }
}
