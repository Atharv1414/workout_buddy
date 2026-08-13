import 'package:shared_preferences/shared_preferences.dart';

import '../models/exercise.dart';

class HistoryRepository {
  static const _key = 'workout_history_v1';
  final SharedPreferencesAsync _preferences = SharedPreferencesAsync();

  Future<List<WorkoutRecord>> load() async {
    final values = await _preferences.getStringList(_key) ?? const [];
    return values.map(WorkoutRecord.decode).whereType<WorkoutRecord>().toList()
      ..sort((a, b) => b.completedAt.compareTo(a.completedAt));
  }

  Future<void> save(WorkoutRecord record) async {
    final current = await _preferences.getStringList(_key) ?? <String>[];
    await _preferences.setStringList(
      _key,
      [record.encode(), ...current].take(100).toList(),
    );
  }
}
