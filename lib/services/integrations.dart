import 'dart:io';

import 'package:health/health.dart';

abstract interface class WorkoutIntegration {
  Future<bool> exportWorkout({
    required String exercise,
    required int reps,
    required Duration duration,
  });
}

class DeviceHealthIntegration implements WorkoutIntegration {
  DeviceHealthIntegration({Health? health}) : _health = health ?? Health();

  final Health _health;

  Future<bool> authorize() async {
    if (!Platform.isIOS && !Platform.isAndroid) return false;
    await _health.configure();
    if (Platform.isAndroid && !await _health.isHealthConnectAvailable()) {
      throw StateError('Health Connect is not available on this device.');
    }
    return _health.requestAuthorization(
      const [HealthDataType.WORKOUT],
      permissions: const [HealthDataAccess.WRITE],
    );
  }

  @override
  Future<bool> exportWorkout({
    required String exercise,
    required int reps,
    required Duration duration,
  }) async {
    await _health.configure();
    if (Platform.isAndroid && !await _health.isHealthConnectAvailable()) {
      throw StateError('Health Connect is not available on this device.');
    }
    final end = DateTime.now();
    final safeDuration = duration.inSeconds > 0
        ? duration
        : const Duration(seconds: 1);
    return _health.writeWorkoutData(
      activityType: HealthWorkoutActivityType.OTHER,
      title: 'Workout Buddy: $exercise ($reps reps)',
      start: end.subtract(safeDuration),
      end: end,
      recordingMethod: RecordingMethod.manual,
    );
  }
}

/// Technogym credentials belong on a backend and are issued only after partner
/// approval. This adapter fails closed until that backend is available.
class TechnogymIntegration implements WorkoutIntegration {
  const TechnogymIntegration();

  @override
  Future<bool> exportWorkout({
    required String exercise,
    required int reps,
    required Duration duration,
  }) {
    throw UnsupportedError(
      'Technogym sync requires approved partner credentials and a secure backend.',
    );
  }
}
