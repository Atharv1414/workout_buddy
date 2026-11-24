import 'dart:convert';

import 'package:test/test.dart';
import 'package:workout_buddy/services/gym_license.dart';
import 'package:workout_buddy/services/license_keys.dart';

// Test-only key pair; the real private key never lives in the repo.
final seed = List<int>.generate(32, (i) => i);
final otherSeed = List<int>.generate(32, (i) => 255 - i);

GymLicense gym({
  String id = 'ironworks',
  String name = 'Ironworks Gym',
  int color = 0xFF7A00,
  GymPlan plan = GymPlan.monthly,
  DateTime? through,
}) => GymLicense(
  gymId: id,
  gymName: name,
  color: color,
  plan: plan,
  paidThrough: through ?? DateTime.utc(2027, 1, 31),
);

void main() {
  late LicenseVerifier verifier;
  setUpAll(() async {
    verifier = LicenseVerifier(await publicKeyFor(seed));
  });

  group('codes', () {
    test('a signed code round-trips', () async {
      final code = await issueCode(gym(plan: GymPlan.annual), seed);
      expect(code, startsWith('WB1.'));
      final license = await verifier.verify(code);
      expect(license.gymId, 'ironworks');
      expect(license.gymName, 'Ironworks Gym');
      expect(license.color, 0xFF7A00);
      expect(license.plan, GymPlan.annual);
      expect(license.paidThrough, DateTime.utc(2027, 1, 31));
    });

    test('accepts the activation link and pasted whitespace', () async {
      final code = await issueCode(gym(), seed);
      final link = activationLink(code).toString();
      expect(link, startsWith('workoutbuddy://activate?code=WB1.'));
      expect((await verifier.verify(link)).gymId, 'ironworks');
      final wrapped = '  ${code.substring(0, 40)}\n${code.substring(40)}\n';
      expect((await verifier.verify(wrapped)).gymId, 'ironworks');
    });

    test('rejects an edited payload with the original signature', () async {
      final code = await issueCode(gym(), seed);
      final parts = code.split('.');
      final forged = base64Url
          .encode(
            utf8.encode(
              jsonEncode(gym(through: DateTime.utc(2099, 12, 31)).toJson()),
            ),
          )
          .replaceAll('=', '');
      expect(
        () => verifier.verify('WB1.$forged.${parts[2]}'),
        throwsA(isA<LicenseException>()),
      );
    });

    test('rejects a code signed with a different key', () async {
      final code = await issueCode(gym(), otherSeed);
      expect(
        () => verifier.verify(code),
        throwsA(
          isA<LicenseException>().having(
            (e) => e.message,
            'message',
            contains('not valid'),
          ),
        ),
      );
    });

    test('rejects junk', () async {
      for (final input in ['', 'hello', 'WB1.abc', 'WB2.a.b', 'WB1.%%%.%%%']) {
        await expectLater(
          verifier.verify(input),
          throwsA(isA<LicenseException>()),
          reason: input,
        );
      }
    });

    test('the built-in public key is a 32-byte Ed25519 key', () {
      expect(base64.decode(gymCodePublicKey), hasLength(32));
    });
  });

  group('payload', () {
    final valid = gym().toJson();

    test('rejects unknown versions and missing or bad fields', () {
      for (final bad in <Map<String, Object>>[
        {...valid, 'v': 2},
        {...valid}..remove('n'),
        {...valid, 'n': '  '},
        {...valid, 'c': 'orange'},
        {...valid, 'p': 'weekly'},
        {...valid, 'to': '2027-02-30'},
        {...valid, 'to': '31/01/2027'},
      ]) {
        expect(
          () => GymLicense.fromJson(bad),
          throwsA(isA<LicenseException>()),
          reason: '$bad',
        );
      }
    });

    test('colour parsing', () {
      expect(parseColorHex('FF7A00'), 0xFF7A00);
      expect(parseColorHex('#0a0b0c'), 0x0A0B0C);
      expect(parseColorHex('FFF'), isNull);
      expect(gym(color: 0x00AB12).colorHex, '00AB12');
    });
  });

  group('status', () {
    final license = gym(through: DateTime.utc(2027, 1, 31));

    test('active through the last paid day', () {
      expect(license.statusAt(DateTime.utc(2027, 1, 1)), LicenseStatus.active);
      expect(
        license.statusAt(DateTime.utc(2027, 1, 31, 23, 59)),
        LicenseStatus.active,
      );
    });

    test('then $graceDays days of grace, then expired', () {
      expect(license.statusAt(DateTime.utc(2027, 2, 1)), LicenseStatus.grace);
      expect(
        license.statusAt(DateTime.utc(2027, 2, 7, 23, 59)),
        LicenseStatus.grace,
      );
      expect(license.statusAt(DateTime.utc(2027, 2, 8)), LicenseStatus.expired);
    });

    test('days left counts today', () {
      expect(license.daysLeftAt(DateTime.utc(2027, 1, 31, 12)), 1);
      expect(license.daysLeftAt(DateTime.utc(2027, 1, 1)), 31);
      expect(license.daysLeftAt(DateTime.utc(2027, 3, 1)), 0);
    });
  });

  group('chooseLicense', () {
    final jan = gym(through: DateTime.utc(2027, 1, 31));
    final feb = gym(through: DateTime.utc(2027, 2, 28));

    test('first activation and renewals take the new code', () {
      expect(chooseLicense(null, jan), same(jan));
      expect(chooseLicense(jan, feb), same(feb));
    });

    test('an older code for the same gym does not shorten access', () {
      expect(chooseLicense(feb, jan), same(feb));
    });

    test('a different gym always replaces the current one', () {
      final other = gym(id: 'pulse', through: DateTime.utc(2026, 12, 31));
      expect(chooseLicense(feb, other), same(other));
    });
  });

  test('formatDay', () {
    expect(formatDay(DateTime.utc(2027, 12, 31)), '31 Dec 2027');
    expect(formatDay(DateTime.utc(2027, 1, 5)), '5 Jan 2027');
  });
}
