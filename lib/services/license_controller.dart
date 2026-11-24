import 'dart:convert';

import 'package:flutter/widgets.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'gym_license.dart';
import 'license_keys.dart';

/// Holds the member's gym subscription and keeps it on the device.
class LicenseController extends ChangeNotifier {
  LicenseController({LicenseVerifier? verifier})
    : _verifier = verifier ?? LicenseVerifier(base64.decode(gymCodePublicKey));

  static const _codeKey = 'gym_code_v1';
  final LicenseVerifier _verifier;
  final SharedPreferencesAsync _preferences = SharedPreferencesAsync();

  GymLicense? _license;
  GymLicense? get license => _license;

  LicenseStatus? get status => _license?.statusAt(DateTime.now());

  /// Whether the app's workouts are available.
  bool get unlocked =>
      status == LicenseStatus.active || status == LicenseStatus.grace;

  /// Restores the saved code. A saved code that no longer verifies (for
  /// example after a signing key change) is dropped.
  Future<void> load() async {
    final code = await _preferences.getString(_codeKey);
    if (code == null) return;
    try {
      _license = await _verifier.verify(code);
    } on LicenseException {
      await _preferences.remove(_codeKey);
    }
    notifyListeners();
  }

  /// Verifies and saves a pasted code or activation link. Returns the
  /// licence now in use. Throws [LicenseException] with a message for the
  /// member if the code is invalid or its paid period is over.
  Future<GymLicense> activate(String input) async {
    final incoming = await _verifier.verify(input);
    final kept = chooseLicense(_license, incoming);
    if (identical(kept, _license)) return kept;
    if (incoming.statusAt(DateTime.now()) == LicenseStatus.expired) {
      throw LicenseException(
        "${incoming.gymName}'s access ended on "
        '${formatDay(incoming.paidThrough)}. Ask your gym for a new code.',
      );
    }
    await _preferences.setString(_codeKey, extractCode(input));
    _license = incoming;
    notifyListeners();
    return incoming;
  }

  Future<void> remove() async {
    await _preferences.remove(_codeKey);
    _license = null;
    notifyListeners();
  }
}

/// Makes the [LicenseController] available below the app root and rebuilds
/// dependants when the subscription changes.
class LicenseScope extends InheritedNotifier<LicenseController> {
  const LicenseScope({
    super.key,
    required LicenseController controller,
    required super.child,
  }) : super(notifier: controller);

  static LicenseController of(BuildContext context) =>
      context.dependOnInheritedWidgetOfExactType<LicenseScope>()!.notifier!;
}
