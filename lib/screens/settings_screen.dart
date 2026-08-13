import 'dart:io';

import 'package:flutter/material.dart';

import '../services/app_preferences.dart';
import '../services/integrations.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _preferences = AppPreferences();
  final _health = DeviceHealthIntegration();
  bool _leftSide = true;
  bool _healthSync = false;
  bool _requestingHealth = false;

  @override
  void initState() {
    super.initState();
    _preferences.preferredLeftSide().then((value) {
      if (mounted) setState(() => _leftSide = value);
    });
    _preferences.healthSyncEnabled().then((value) {
      if (mounted) setState(() => _healthSync = value);
    });
  }

  Future<void> _setHealthSync(bool value) async {
    if (!value) {
      setState(() => _healthSync = false);
      await _preferences.setHealthSyncEnabled(false);
      return;
    }
    setState(() => _requestingHealth = true);
    try {
      final authorized = await _health.authorize();
      await _preferences.setHealthSyncEnabled(authorized);
      if (mounted) setState(() => _healthSync = authorized);
      if (!authorized && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Health access was not granted.')),
        );
      }
    } catch (error) {
      await _preferences.setHealthSyncEnabled(false);
      if (mounted) {
        setState(() => _healthSync = false);
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Could not connect: $error')));
      }
    } finally {
      if (mounted) setState(() => _requestingHealth = false);
    }
  }

  @override
  Widget build(BuildContext context) => ListView(
    padding: const EdgeInsets.all(20),
    children: [
      Text(
        'Settings',
        style: Theme.of(
          context,
        ).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w800),
      ),
      const SizedBox(height: 22),
      Text('Training', style: Theme.of(context).textTheme.titleMedium),
      const SizedBox(height: 8),
      Card(
        child: SwitchListTile(
          value: _leftSide,
          onChanged: (value) async {
            setState(() => _leftSide = value);
            await _preferences.setPreferredLeftSide(value);
          },
          title: const Text('Track left side by default'),
          subtitle: const Text('You can change sides during any workout.'),
          secondary: const Icon(Icons.accessibility_new),
        ),
      ),
      const SizedBox(height: 18),
      Text('Connections', style: Theme.of(context).textTheme.titleMedium),
      const SizedBox(height: 8),
      Card(
        child: SwitchListTile(
          value: _healthSync,
          onChanged: _requestingHealth ? null : _setHealthSync,
          title: Text(Platform.isIOS ? 'Apple Health' : 'Health Connect'),
          subtitle: Text(
            _requestingHealth
                ? 'Requesting access…'
                : 'Save completed workouts automatically',
          ),
          secondary: _requestingHealth
              ? const SizedBox.square(
                  dimension: 24,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.favorite_outline),
        ),
      ),
      const _ConnectionTile(
        icon: Icons.fitness_center,
        title: 'Technogym',
        subtitle: 'Requires approved partner credentials',
        enabled: false,
      ),
      const SizedBox(height: 18),
      Text('About', style: Theme.of(context).textTheme.titleMedium),
      const SizedBox(height: 8),
      Card(
        child: ListTile(
          leading: Icon(Icons.shield_outlined),
          title: const Text('Privacy'),
          subtitle: const Text('Camera frames stay on this device.'),
          onTap: () => showDialog<void>(
            context: context,
            builder: (_) => AlertDialog(
              title: const Text('Privacy at a glance'),
              content: const Text(
                'Pose detection runs on this device. Workout summaries remain local unless you explicitly enable Apple Health or Health Connect. Workout Buddy does not sell personal data.',
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Done'),
                ),
              ],
            ),
          ),
        ),
      ),
      const Card(
        child: ListTile(
          leading: Icon(Icons.info_outline),
          title: Text('Workout Buddy'),
          subtitle: Text('Version 1.0.0'),
        ),
      ),
      const Padding(
        padding: EdgeInsets.all(16),
        child: Text(
          'Workout Buddy is a fitness aid, not medical advice.',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.white54),
        ),
      ),
    ],
  );
}

class _ConnectionTile extends StatelessWidget {
  const _ConnectionTile({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.enabled,
  });
  final IconData icon;
  final String title;
  final String subtitle;
  final bool enabled;

  @override
  Widget build(BuildContext context) => Card(
    child: ListTile(
      leading: Icon(icon),
      title: Text(title),
      subtitle: Text(subtitle),
      trailing: Chip(label: Text(enabled ? 'Available' : 'Pending')),
      onTap: () => showDialog<void>(
        context: context,
        builder: (_) => AlertDialog(
          title: Text(title),
          content: Text(
            enabled
                ? 'This connection is available.'
                : 'Technogym must approve Workout Buddy and issue server credentials before this connection can be enabled.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('OK'),
            ),
          ],
        ),
      ),
    ),
  );
}
