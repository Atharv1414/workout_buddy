import 'package:shared_preferences/shared_preferences.dart';

class AppPreferences {
  static const _onboardingKey = 'onboarding_complete_v1';
  static const _sideKey = 'preferred_side_v1';
  static const _voiceKey = 'voice_cues_v1';
  static const _healthSyncKey = 'health_sync_enabled_v1';

  final SharedPreferencesAsync _preferences = SharedPreferencesAsync();

  Future<bool> onboardingComplete() async =>
      await _preferences.getBool(_onboardingKey) ?? false;

  Future<void> completeOnboarding() =>
      _preferences.setBool(_onboardingKey, true);

  Future<bool> preferredLeftSide() async =>
      await _preferences.getBool(_sideKey) ?? true;

  Future<void> setPreferredLeftSide(bool value) =>
      _preferences.setBool(_sideKey, value);

  Future<bool> voiceCuesEnabled() async =>
      await _preferences.getBool(_voiceKey) ?? true;

  Future<void> setVoiceCuesEnabled(bool value) =>
      _preferences.setBool(_voiceKey, value);

  Future<bool> healthSyncEnabled() async =>
      await _preferences.getBool(_healthSyncKey) ?? false;

  Future<void> setHealthSyncEnabled(bool value) =>
      _preferences.setBool(_healthSyncKey, value);
}
