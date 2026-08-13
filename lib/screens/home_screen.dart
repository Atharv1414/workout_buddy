import 'package:flutter/material.dart';

import '../models/exercise.dart';
import '../services/history_repository.dart';
import 'settings_screen.dart';
import 'workout_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _history = HistoryRepository();
  int _page = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: [_trainPage(), _progressPage(), const SettingsScreen()][_page],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _page,
        onDestinationSelected: (value) => setState(() => _page = value),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.fitness_center),
            label: 'Train',
          ),
          NavigationDestination(icon: Icon(Icons.insights), label: 'Progress'),
          NavigationDestination(
            icon: Icon(Icons.settings_outlined),
            label: 'Settings',
          ),
        ],
      ),
    );
  }

  Widget _trainPage() {
    return ListView(
      padding: const EdgeInsets.fromLTRB(20, 24, 20, 28),
      children: [
        Text(
          'Workout Buddy',
          style: Theme.of(
            context,
          ).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w800),
        ),
        const SizedBox(height: 6),
        Text(
          'Your private, real-time form coach',
          style: Theme.of(
            context,
          ).textTheme.bodyLarge?.copyWith(color: Colors.white60),
        ),
        const SizedBox(height: 28),
        Container(
          padding: const EdgeInsets.all(22),
          decoration: BoxDecoration(
            gradient: const LinearGradient(
              colors: [Color(0xFF1D5139), Color(0xFF1A2822)],
            ),
            borderRadius: BorderRadius.circular(24),
          ),
          child: const Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                Icons.visibility_outlined,
                size: 32,
                color: Color(0xFF71E6A5),
              ),
              SizedBox(height: 18),
              Text(
                'Move with confidence',
                style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800),
              ),
              SizedBox(height: 8),
              Text(
                'Pose analysis happens on your phone. Video is never uploaded.',
                style: TextStyle(color: Colors.white70, height: 1.4),
              ),
            ],
          ),
        ),
        const SizedBox(height: 28),
        Text(
          'Choose a workout',
          style: Theme.of(
            context,
          ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 12),
        ...Exercise.all.map(
          (exercise) => Card(
            margin: const EdgeInsets.only(bottom: 12),
            clipBehavior: Clip.antiAlias,
            child: ListTile(
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 18,
                vertical: 11,
              ),
              leading: CircleAvatar(
                backgroundColor: Colors.white10,
                child: Text(exercise.emoji),
              ),
              title: Text(
                exercise.name,
                style: const TextStyle(fontWeight: FontWeight.w700),
              ),
              subtitle: Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(exercise.subtitle),
              ),
              trailing: const Icon(Icons.arrow_forward_ios, size: 16),
              onTap: () async {
                await Navigator.of(context).push(
                  MaterialPageRoute(
                    builder: (_) => WorkoutScreen(exercise: exercise),
                  ),
                );
                if (mounted) setState(() {});
              },
            ),
          ),
        ),
        const SizedBox(height: 14),
        const _IntegrationCard(),
      ],
    );
  }

  Widget _progressPage() {
    return FutureBuilder<List<WorkoutRecord>>(
      future: _history.load(),
      builder: (context, snapshot) {
        final records = snapshot.data ?? const <WorkoutRecord>[];
        final total = records.fold<int>(0, (sum, item) => sum + item.reps);
        return ListView(
          padding: const EdgeInsets.all(20),
          children: [
            Text(
              'Progress',
              style: Theme.of(
                context,
              ).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 20),
            Row(
              children: [
                Expanded(
                  child: _Stat(label: 'Workouts', value: '${records.length}'),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: _Stat(label: 'Total reps', value: '$total'),
                ),
              ],
            ),
            const SizedBox(height: 24),
            Text(
              'Recent workouts',
              style: Theme.of(
                context,
              ).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 10),
            if (records.isEmpty)
              const Card(
                child: Padding(
                  padding: EdgeInsets.all(24),
                  child: Text('Complete your first workout to see it here.'),
                ),
              ),
            ...records.map(
              (record) => Card(
                child: ListTile(
                  leading: const Icon(
                    Icons.check_circle,
                    color: Color(0xFF71E6A5),
                  ),
                  title: Text(record.exercise),
                  subtitle: Text('${record.reps} reps • ${record.seconds}s'),
                  trailing: Text(
                    '${record.completedAt.day}/${record.completedAt.month}',
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _Stat extends StatelessWidget {
  const _Stat({required this.label, required this.value});
  final String label;
  final String value;
  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            value,
            style: const TextStyle(
              fontSize: 30,
              fontWeight: FontWeight.w800,
              color: Color(0xFF71E6A5),
            ),
          ),
          Text(label, style: const TextStyle(color: Colors.white60)),
        ],
      ),
    ),
  );
}

class _IntegrationCard extends StatelessWidget {
  const _IntegrationCard();
  @override
  Widget build(BuildContext context) => Card(
    child: Padding(
      padding: const EdgeInsets.all(18),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.sync, color: Color(0xFF71E6A5)),
              SizedBox(width: 10),
              Text(
                'Connected fitness',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 17),
              ),
            ],
          ),
          const SizedBox(height: 10),
          const Text(
            'Save workouts to Apple Health or Health Connect. Technogym requires partner approval.',
            style: TextStyle(color: Colors.white60, height: 1.4),
          ),
          const SizedBox(height: 8),
          TextButton(
            onPressed: () => showDialog<void>(
              context: context,
              builder: (_) => const AlertDialog(
                title: Text('Integration setup'),
                content: Text(
                  'Health permissions are enabled during store signing. Technogym access requires an approved API key and a secure backend; keys are never stored in this app.',
                ),
              ),
            ),
            child: const Text('How connections work'),
          ),
        ],
      ),
    ),
  );
}
