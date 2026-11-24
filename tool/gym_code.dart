// Creates gym access codes for Workout Buddy.
//
//   dart run tool/gym_code.dart keygen
//   dart run tool/gym_code.dart issue --id ironworks --name "Ironworks Gym" \
//       --color FF7A00 --plan annual --until 2027-11-30
//
// The private signing key lives OUTSIDE the repo (default
// ~/.workout_buddy/signing_key.json). Anyone with that file can create codes,
// so back it up somewhere safe and never commit or share it. See
// docs/GYM_SUBSCRIPTIONS.md.
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:workout_buddy/services/gym_license.dart';
import 'package:workout_buddy/services/license_keys.dart';

String get _defaultKeyPath =>
    '${Platform.environment['HOME']}/.workout_buddy/signing_key.json';

Future<void> main(List<String> args) async {
  if (args.isEmpty) return _usage();
  final options = _options(args.skip(1).toList());
  final keyPath = options['key'] ?? _defaultKeyPath;
  switch (args.first) {
    case 'keygen':
      await _keygen(keyPath);
    case 'issue':
      await _issue(options, keyPath);
    default:
      _usage();
  }
}

Future<void> _keygen(String keyPath) async {
  final file = File(keyPath);
  if (file.existsSync()) {
    _fail(
      'A signing key already exists at $keyPath.\n'
      'Replacing it would make every code you have issued invalid in the next '
      'app release. Move it away yourself first if you really mean to.',
    );
  }
  final random = Random.secure();
  final seed = List<int>.generate(32, (_) => random.nextInt(256));
  final publicKey = await publicKeyFor(seed);
  file.parent.createSync(recursive: true);
  file.writeAsStringSync(
    const JsonEncoder.withIndent('  ').convert({
      'note': 'Workout Buddy gym-code signing key. Keep secret. Back it up.',
      'seed': base64.encode(seed),
      'publicKey': base64.encode(publicKey),
    }),
  );
  if (!Platform.isWindows) {
    Process.runSync('chmod', ['600', keyPath]);
  }
  stdout.writeln('Signing key saved to $keyPath');
  stdout.writeln('Put this public key in lib/services/license_keys.dart:');
  stdout.writeln("  '${base64.encode(publicKey)}'");
}

Future<void> _issue(Map<String, String> o, String keyPath) async {
  final id = o['id'];
  final name = o['name'];
  final color = parseColorHex(o['color'] ?? '71E6A5');
  final plan = GymPlan.values.where((p) => p.name == o['plan']).firstOrNull;
  final until = DateTime.tryParse(o['until'] ?? '');
  if (id == null || !RegExp(r'^[a-z0-9-]{2,40}$').hasMatch(id)) {
    _fail('--id is required: 2-40 lowercase letters, digits or dashes.');
  }
  if (name == null || name.trim().isEmpty) _fail('--name is required.');
  if (color == null) _fail('--color must be 6 hex digits, e.g. FF7A00.');
  if (plan == null) _fail('--plan must be monthly or annual.');
  if (until == null) _fail('--until must be a date like 2027-11-30.');

  final file = File(keyPath);
  if (!file.existsSync()) {
    _fail('No signing key at $keyPath. Run "keygen" first, or pass --key.');
  }
  final json = jsonDecode(file.readAsStringSync()) as Map<String, Object?>;
  final seed = base64.decode(json['seed']! as String);
  final publicKey = base64.encode(await publicKeyFor(seed));
  if (publicKey != gymCodePublicKey) {
    _fail(
      'This signing key does not match the public key built into the app '
      '(lib/services/license_keys.dart). Codes from it would be rejected.',
    );
  }

  final license = GymLicense(
    gymId: id,
    gymName: name.trim(),
    color: color,
    plan: plan,
    paidThrough: DateTime.utc(until.year, until.month, until.day),
  );
  final code = await issueCode(license, seed);
  // Check it the same way the app will.
  await LicenseVerifier(base64.decode(gymCodePublicKey)).verify(code);

  stdout
    ..writeln('Gym:        ${license.gymName} (${license.gymId})')
    ..writeln('Plan:       ${license.plan.label}')
    ..writeln('Paid until: ${o['until']} (+$graceDays days grace)')
    ..writeln()
    ..writeln('Code (members can paste this):')
    ..writeln(code)
    ..writeln()
    ..writeln("Link (for the gym's app or website button):")
    ..writeln(activationLink(code));
}

Map<String, String> _options(List<String> args) {
  final result = <String, String>{};
  for (var i = 0; i < args.length; i++) {
    final arg = args[i];
    if (!arg.startsWith('--')) _fail('Unexpected argument: $arg');
    if (i + 1 >= args.length) _fail('Missing value for $arg');
    result[arg.substring(2)] = args[++i];
  }
  return result;
}

void _usage() {
  stdout.writeln('''
Usage:
  dart run tool/gym_code.dart keygen [--key path]
  dart run tool/gym_code.dart issue --id <gym-id> --name "<Gym name>"
      --color <RRGGBB> --plan monthly|annual --until <YYYY-MM-DD> [--key path]
''');
  exit(64);
}

Never _fail(String message) {
  stderr.writeln(message);
  exit(1);
}
