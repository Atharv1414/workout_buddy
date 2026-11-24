import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../services/gym_license.dart';
import '../services/license_controller.dart';

/// Shown until the member's gym has activated Workout Buddy, and again once
/// the gym's subscription has run out. Also pushed from Settings to change
/// gym ([standalone]).
class ActivationScreen extends StatefulWidget {
  const ActivationScreen({super.key, this.standalone = false});

  /// True when opened from Settings: shows a back button and closes itself
  /// after a successful activation.
  final bool standalone;

  @override
  State<ActivationScreen> createState() => _ActivationScreenState();
}

class _ActivationScreenState extends State<ActivationScreen> {
  final _code = TextEditingController();
  bool _busy = false;
  String? _error;

  @override
  void dispose() {
    _code.dispose();
    super.dispose();
  }

  Future<void> _paste() async {
    final data = await Clipboard.getData(Clipboard.kTextPlain);
    final text = data?.text;
    if (text == null || text.trim().isEmpty) return;
    setState(() {
      _code.text = text.trim();
      _error = null;
    });
  }

  Future<void> _activate() async {
    final input = _code.text;
    if (input.trim().isEmpty) {
      setState(() => _error = 'Paste the code from your gym first.');
      return;
    }
    setState(() {
      _busy = true;
      _error = null;
    });
    final licenses = LicenseScope.of(context);
    final messenger = ScaffoldMessenger.of(context);
    final navigator = Navigator.of(context);
    try {
      final license = await licenses.activate(input);
      messenger.showSnackBar(
        SnackBar(content: Text('Activated by ${license.gymName}')),
      );
      if (widget.standalone && mounted) navigator.pop();
    } on LicenseException catch (error) {
      if (mounted) setState(() => _error = error.message);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final licenses = LicenseScope.of(context);
    final ended = licenses.license;
    final expired = licenses.status == LicenseStatus.expired;
    final theme = Theme.of(context);
    return Scaffold(
      appBar: widget.standalone ? AppBar(title: const Text('Gym code')) : null,
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 40, 24, 24),
          children: [
            Icon(
              Icons.store_mall_directory_outlined,
              size: 48,
              color: theme.colorScheme.primary,
            ),
            const SizedBox(height: 20),
            Text(
              expired ? 'Your gym access has ended' : 'Activate with your gym',
              style: theme.textTheme.headlineSmall?.copyWith(
                fontWeight: FontWeight.w800,
              ),
            ),
            const SizedBox(height: 10),
            Text(
              expired && ended != null
                  ? "${ended.gymName}'s Workout Buddy access ended on "
                        '${formatDay(ended.paidThrough)}. Ask your gym to '
                        'renew, then open Workout Buddy from their app or '
                        'enter the new code.'
                  : 'Workout Buddy is included with your gym membership. '
                        "Open it from your gym's app, or enter the code your "
                        'gym gave you.',
              style: const TextStyle(color: Colors.white70, height: 1.45),
            ),
            const SizedBox(height: 28),
            TextField(
              controller: _code,
              minLines: 2,
              maxLines: 4,
              autocorrect: false,
              enableSuggestions: false,
              keyboardType: TextInputType.visiblePassword,
              decoration: InputDecoration(
                labelText: 'Gym code',
                hintText: 'WB1.…',
                errorText: _error,
                errorMaxLines: 3,
                border: const OutlineInputBorder(),
                suffixIcon: IconButton(
                  tooltip: 'Paste',
                  icon: const Icon(Icons.content_paste),
                  onPressed: _busy ? null : _paste,
                ),
              ),
              onChanged: (_) {
                if (_error != null) setState(() => _error = null);
              },
            ),
            const SizedBox(height: 18),
            FilledButton(
              onPressed: _busy ? null : _activate,
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(52),
              ),
              child: _busy
                  ? const SizedBox.square(
                      dimension: 22,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Activate'),
            ),
          ],
        ),
      ),
    );
  }
}
