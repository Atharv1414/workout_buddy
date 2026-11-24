import 'dart:convert';

import 'package:cryptography/cryptography.dart';

/// Gym subscriptions are unlocked with an offline, signed access code.
///
/// A code is `WB1.<payload>.<signature>`, both parts base64url without
/// padding. The payload is JSON describing the gym and when its paid period
/// ends; the signature is Ed25519 over the payload bytes, made with a private
/// key that never ships in the app (see `tool/gym_code.dart`). The app only
/// holds the public key, so it can check codes but never create them.
///
/// This file must stay free of Flutter imports: the code generator runs it
/// with plain `dart run`.

const codePrefix = 'WB1';

/// Days the app keeps working after a paid period ends, so a late renewal
/// doesn't lock members out mid-week.
const graceDays = 7;

enum GymPlan {
  monthly('Monthly'),
  annual('Annual');

  const GymPlan(this.label);
  final String label;
}

enum LicenseStatus { active, grace, expired }

class LicenseException implements Exception {
  const LicenseException(this.message);
  final String message;
  @override
  String toString() => message;
}

class GymLicense {
  const GymLicense({
    required this.gymId,
    required this.gymName,
    required this.color,
    required this.plan,
    required this.paidThrough,
  });

  /// Short stable identifier, e.g. `ironworks-pune`.
  final String gymId;
  final String gymName;

  /// Brand colour as 0xRRGGBB.
  final int color;
  final GymPlan plan;

  /// Last day (UTC calendar date) the gym has paid for. Access lasts to the
  /// end of that day, then [graceDays] more.
  final DateTime paidThrough;

  DateTime get _accessEnds =>
      DateTime.utc(paidThrough.year, paidThrough.month, paidThrough.day + 1);

  DateTime get graceEnds => _accessEnds.add(const Duration(days: graceDays));

  LicenseStatus statusAt(DateTime now) {
    final utc = now.toUtc();
    if (utc.isBefore(_accessEnds)) return LicenseStatus.active;
    if (utc.isBefore(graceEnds)) return LicenseStatus.grace;
    return LicenseStatus.expired;
  }

  /// Whole days of paid access left, counting today; 0 once it has ended.
  int daysLeftAt(DateTime now) {
    final utc = now.toUtc();
    final today = DateTime.utc(utc.year, utc.month, utc.day);
    final days = _accessEnds.difference(today).inDays;
    return days < 0 ? 0 : days;
  }

  String get colorHex => color.toRadixString(16).padLeft(6, '0').toUpperCase();

  Map<String, Object> toJson() => {
    'v': 1,
    'id': gymId,
    'n': gymName,
    'c': colorHex,
    'p': plan.name,
    'to': _date(paidThrough),
  };

  static GymLicense fromJson(Object? json) {
    if (json is! Map<String, Object?> || json['v'] != 1) {
      throw const LicenseException(
        'This code is for a newer version of the app.',
      );
    }
    final id = json['id'];
    final name = json['n'];
    final color = json['c'];
    final plan = json['p'];
    final to = json['to'];
    if (id is! String ||
        id.isEmpty ||
        name is! String ||
        name.trim().isEmpty ||
        color is! String ||
        plan is! String ||
        to is! String) {
      throw const LicenseException('This code is incomplete.');
    }
    final parsedColor = parseColorHex(color);
    final parsedPlan = GymPlan.values.where((p) => p.name == plan).firstOrNull;
    final parsedDate = _parseDate(to);
    if (parsedColor == null || parsedPlan == null || parsedDate == null) {
      throw const LicenseException('This code is incomplete.');
    }
    return GymLicense(
      gymId: id,
      gymName: name.trim(),
      color: parsedColor,
      plan: parsedPlan,
      paidThrough: parsedDate,
    );
  }
}

/// Parses `RRGGBB` or `#RRGGBB`; null if it isn't a 6-digit hex colour.
int? parseColorHex(String value) {
  final hex = value.startsWith('#') ? value.substring(1) : value;
  if (!RegExp(r'^[0-9a-fA-F]{6}$').hasMatch(hex)) return null;
  return int.parse(hex, radix: 16);
}

String _date(DateTime d) =>
    '${d.year.toString().padLeft(4, '0')}-'
    '${d.month.toString().padLeft(2, '0')}-'
    '${d.day.toString().padLeft(2, '0')}';

DateTime? _parseDate(String value) {
  final match = RegExp(r'^(\d{4})-(\d{2})-(\d{2})$').firstMatch(value);
  if (match == null) return null;
  final year = int.parse(match[1]!);
  final month = int.parse(match[2]!);
  final day = int.parse(match[3]!);
  final date = DateTime.utc(year, month, day);
  // Reject dates DateTime silently rolls over, like 2027-02-30.
  if (date.month != month || date.day != day) return null;
  return date;
}

String _b64(List<int> bytes) => base64Url.encode(bytes).replaceAll('=', '');

List<int> _unb64(String value) => base64Url.decode(base64Url.normalize(value));

/// Pulls the code out of whatever the member pasted or the link carried:
/// a bare code, or a `workoutbuddy://activate?code=…` link.
String extractCode(String input) {
  final trimmed = input.trim();
  final uri = Uri.tryParse(trimmed);
  final fromLink = uri?.queryParameters['code'];
  final code = (fromLink ?? trimmed).replaceAll(RegExp(r'\s'), '');
  return code;
}

/// Builds the link a gym puts behind a button in its own app.
Uri activationLink(String code) => Uri(
  scheme: 'workoutbuddy',
  host: 'activate',
  queryParameters: {'code': code},
);

final _ed25519 = Ed25519();

/// Signs [license]. Only used by the code generator and tests.
Future<String> issueCode(GymLicense license, List<int> privateSeed) async {
  final keyPair = await _ed25519.newKeyPairFromSeed(privateSeed);
  final payload = utf8.encode(jsonEncode(license.toJson()));
  final signature = await _ed25519.sign(payload, keyPair: keyPair);
  return '$codePrefix.${_b64(payload)}.${_b64(signature.bytes)}';
}

/// Returns the 32-byte public key for a 32-byte private seed.
Future<List<int>> publicKeyFor(List<int> privateSeed) async {
  final keyPair = await _ed25519.newKeyPairFromSeed(privateSeed);
  final publicKey = await keyPair.extractPublicKey();
  return publicKey.bytes;
}

class LicenseVerifier {
  const LicenseVerifier(this.publicKey);
  final List<int> publicKey;

  /// Checks the signature and decodes the gym. Throws [LicenseException]
  /// with a member-friendly message if the code is wrong. Does not check
  /// the date; use [GymLicense.statusAt] for that.
  Future<GymLicense> verify(String input) async {
    final code = extractCode(input);
    final parts = code.split('.');
    if (parts.length != 3 || parts[0] != codePrefix) {
      throw const LicenseException(
        "That doesn't look like a Workout Buddy gym code.",
      );
    }
    final List<int> payload;
    final List<int> signature;
    try {
      payload = _unb64(parts[1]);
      signature = _unb64(parts[2]);
    } on FormatException {
      throw const LicenseException(
        'This code is damaged. Copy it again or use the link from your gym.',
      );
    }
    final valid =
        signature.length == 64 &&
        await _ed25519.verify(
          payload,
          signature: Signature(
            signature,
            publicKey: SimplePublicKey(publicKey, type: KeyPairType.ed25519),
          ),
        );
    if (!valid) {
      throw const LicenseException(
        'This code is not valid. Check it with your gym.',
      );
    }
    final Object? json;
    try {
      json = jsonDecode(utf8.decode(payload));
    } on FormatException {
      throw const LicenseException('This code is incomplete.');
    }
    return GymLicense.fromJson(json);
  }
}

/// Decides which licence to keep when a member activates [incoming].
///
/// Gyms keep handing out their current code (e.g. behind a button in the
/// gym's app), so an older code for the same gym must never cut short a
/// newer one the member already has. A code for a different gym always wins:
/// the member has moved gyms.
GymLicense chooseLicense(GymLicense? current, GymLicense incoming) {
  if (current != null &&
      current.gymId == incoming.gymId &&
      current.paidThrough.isAfter(incoming.paidThrough)) {
    return current;
  }
  return incoming;
}

const _months = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];

/// `31 Dec 2027`.
String formatDay(DateTime date) =>
    '${date.day} ${_months[date.month - 1]} ${date.year}';
