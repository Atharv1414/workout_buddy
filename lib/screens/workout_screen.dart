import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import '../models/exercise.dart';
import '../services/history_repository.dart';
import '../services/app_preferences.dart';
import '../services/frame_rotation.dart';
import '../services/integrations.dart';
import '../services/rep_counter.dart';

const _deviceRotation = <DeviceOrientation, int>{
  DeviceOrientation.portraitUp: 0,
  DeviceOrientation.landscapeLeft: 90,
  DeviceOrientation.portraitDown: 180,
  DeviceOrientation.landscapeRight: 270,
};

String _cameraErrorMessage(Object error) {
  if (error is CameraException) {
    switch (error.code) {
      case 'CameraAccessDenied':
      case 'CameraAccessDeniedWithoutPrompt':
      case 'CameraAccessRestricted':
        return 'Camera access is turned off. Allow it for Workout Buddy in '
            'Settings, then try again.';
    }
  }
  if (error is StateError) return error.message;
  return 'The camera could not start. Close other apps that may be using it, '
      'then try again.';
}

class WorkoutScreen extends StatefulWidget {
  const WorkoutScreen({super.key, required this.exercise});
  final Exercise exercise;

  @override
  State<WorkoutScreen> createState() => _WorkoutScreenState();
}

class _WorkoutScreenState extends State<WorkoutScreen>
    with WidgetsBindingObserver {
  CameraController? _camera;
  late final PoseDetector _detector;
  late RepCounter _counter;
  final _history = HistoryRepository();
  final _preferences = AppPreferences();
  final _health = DeviceHealthIntegration();
  final _watch = Stopwatch();
  Timer? _ticker;
  bool _processing = false;
  bool _streaming = false;
  bool _opening = false;
  bool _inBackground = false;
  bool _finished = false;
  bool _leftSide = true;
  String _feedback = 'Step back until your full body is visible';
  double? _angle;
  int _seconds = 0;
  Object? _error;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _counter = RepCounter(widget.exercise);
    _detector = PoseDetector(
      options: PoseDetectorOptions(mode: PoseDetectionMode.stream),
    );
    _start();
  }

  Future<void> _start() async {
    _leftSide = await _preferences.preferredLeftSide();
    await _startSession();
  }

  /// Opens the camera, then runs the workout timer while it is open.
  Future<void> _startSession() async {
    if (_finished || !await _openCamera()) return;
    _watch.start();
    _ticker ??= Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) setState(() => _seconds = _watch.elapsed.inSeconds);
    });
  }

  Future<bool> _openCamera() async {
    if (_camera != null) return true;
    if (_opening) return false;
    _opening = true;
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw StateError('No camera was found.');
      final selected =
          cameras
              .where((item) => item.lensDirection == CameraLensDirection.front)
              .firstOrNull ??
          cameras.first;
      final controller = CameraController(
        selected,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );
      await controller.initialize();
      // The app may have been backgrounded or closed while the camera opened.
      if (!mounted || _inBackground || _finished) {
        await controller.dispose();
        return false;
      }
      _camera = controller;
      await _startStream();
      if (mounted) setState(() {});
      return true;
    } catch (error) {
      if (mounted) setState(() => _error = error);
      return false;
    } finally {
      _opening = false;
    }
  }

  /// Releases the camera so other apps (and the OS) can use it. The camera
  /// plugin recommends this whenever the app becomes inactive.
  Future<void> _closeCamera() async {
    final camera = _camera;
    if (camera == null) return;
    _camera = null;
    _streaming = false;
    if (mounted) setState(() {});
    await camera.dispose();
  }

  Future<void> _startStream() async {
    final camera = _camera;
    if (camera == null || !camera.value.isInitialized || _streaming) return;
    await camera.startImageStream(_processFrame);
    _streaming = true;
  }

  Future<void> _stopStream() async {
    final camera = _camera;
    if (camera == null || !_streaming) return;
    await camera.stopImageStream();
    _streaming = false;
  }

  Future<void> _retry() async {
    setState(() => _error = null);
    _ticker?.cancel();
    _ticker = null;
    _watch
      ..stop()
      ..reset();
    _seconds = 0;
    await _closeCamera();
    await _startSession();
  }

  void _reset() {
    setState(() {
      _counter = RepCounter(widget.exercise);
      _watch.reset();
      _seconds = 0;
    });
  }

  Future<void> _processFrame(CameraImage frame) async {
    final camera = _camera;
    if (_processing || !mounted || camera == null) return;
    _processing = true;
    try {
      final degrees = frameRotationDegrees(
        isAndroid: Platform.isAndroid,
        sensorOrientation: camera.description.sensorOrientation,
        frontCamera:
            camera.description.lensDirection == CameraLensDirection.front,
        deviceRotation: _deviceRotation[camera.value.deviceOrientation],
      );
      final rotation = degrees == null
          ? null
          : InputImageRotationValue.fromRawValue(degrees);
      final format = InputImageFormatValue.fromRawValue(frame.format.raw);
      if (rotation == null || format == null || frame.planes.length != 1) {
        return;
      }
      final input = InputImage.fromBytes(
        bytes: frame.planes.first.bytes,
        metadata: InputImageMetadata(
          size: ui.Size(frame.width.toDouble(), frame.height.toDouble()),
          rotation: rotation,
          format: format,
          bytesPerRow: frame.planes.first.bytesPerRow,
        ),
      );
      final poses = await _detector.processImage(input);
      if (poses.isEmpty) {
        _setReading(null, 'Step back until your full body is visible');
        return;
      }
      final points = _pointsFor(poses.first);
      if (points == null) {
        _setReading(
          null,
          'Keep your ${_leftSide ? 'left' : 'right'} side visible',
        );
        return;
      }
      final angle = jointAngle(points.$1, points.$2, points.$3);
      final update = _counter.update(angle);
      _setReading(angle, update.feedback);
    } finally {
      _processing = false;
    }
  }

  (JointPoint, JointPoint, JointPoint)? _pointsFor(Pose pose) {
    final left = _leftSide;
    final types = widget.exercise.type == ExerciseType.squat
        ? (
            left ? PoseLandmarkType.leftHip : PoseLandmarkType.rightHip,
            left ? PoseLandmarkType.leftKnee : PoseLandmarkType.rightKnee,
            left ? PoseLandmarkType.leftAnkle : PoseLandmarkType.rightAnkle,
          )
        : (
            left
                ? PoseLandmarkType.leftShoulder
                : PoseLandmarkType.rightShoulder,
            left ? PoseLandmarkType.leftElbow : PoseLandmarkType.rightElbow,
            left ? PoseLandmarkType.leftWrist : PoseLandmarkType.rightWrist,
          );
    final a = pose.landmarks[types.$1];
    final b = pose.landmarks[types.$2];
    final c = pose.landmarks[types.$3];
    if (a == null ||
        b == null ||
        c == null ||
        [a, b, c].any((item) => item.likelihood < .55)) {
      return null;
    }
    return (JointPoint(a.x, a.y), JointPoint(b.x, b.y), JointPoint(c.x, c.y));
  }

  void _setReading(double? angle, String feedback) {
    if (!mounted) return;
    setState(() {
      _angle = angle;
      _feedback = feedback;
    });
  }

  Future<void> _finish() async {
    if (_finished) return;
    _finished = true;
    _watch.stop();
    await _stopStream();
    bool? healthSynced;
    if (_counter.reps > 0) {
      await _history.save(
        WorkoutRecord(
          exercise: widget.exercise.name,
          reps: _counter.reps,
          seconds: _watch.elapsed.inSeconds,
          completedAt: DateTime.now(),
        ),
      );
      if (await _preferences.healthSyncEnabled()) {
        try {
          healthSynced = await _health.exportWorkout(
            exercise: widget.exercise.name,
            reps: _counter.reps,
            duration: _watch.elapsed,
          );
        } catch (_) {
          healthSynced = false;
        }
      }
    }
    if (!mounted) return;
    await showModalBottomSheet<void>(
      context: context,
      isDismissible: false,
      builder: (context) => Padding(
        padding: const EdgeInsets.fromLTRB(24, 24, 24, 38),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.emoji_events, size: 50, color: Color(0xFF71E6A5)),
            const SizedBox(height: 12),
            Text(
              'Workout complete',
              style: Theme.of(
                context,
              ).textTheme.headlineSmall?.copyWith(fontWeight: FontWeight.w800),
            ),
            const SizedBox(height: 8),
            Text(
              '${_counter.reps} reps in ${_watch.elapsed.inSeconds} seconds',
            ),
            if (healthSynced != null) ...[
              const SizedBox(height: 8),
              Text(
                healthSynced
                    ? 'Saved to your health app'
                    : 'Saved locally; health sync was unavailable',
                style: TextStyle(
                  color: healthSynced ? const Color(0xFF71E6A5) : Colors.amber,
                ),
              ),
            ],
            const SizedBox(height: 22),
            SizedBox(
              width: double.infinity,
              child: FilledButton(
                onPressed: () {
                  Navigator.pop(context);
                  Navigator.pop(context);
                },
                child: const Text('Done'),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _tryClose() async {
    if (_counter.reps == 0 || _finished) {
      if (mounted) Navigator.pop(context);
      return;
    }
    final discard = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Discard workout?'),
        content: Text('Your ${_counter.reps} reps have not been saved.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Keep training'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Discard'),
          ),
        ],
      ),
    );
    if (discard == true && mounted) Navigator.pop(context);
  }

  @override
  Future<void> didChangeAppLifecycleState(AppLifecycleState state) async {
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _inBackground = true;
      _watch.stop();
      await _closeCamera();
    } else if (state == AppLifecycleState.resumed) {
      _inBackground = false;
      if (!_finished && _error == null) await _startSession();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _ticker?.cancel();
    _watch.stop();
    _camera?.dispose();
    _detector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final camera = _camera;
    return PopScope(
      canPop: _counter.reps == 0 || _finished,
      onPopInvokedWithResult: (didPop, _) {
        if (!didPop) _tryClose();
      },
      child: Scaffold(
        backgroundColor: Colors.black,
        body: Stack(
          fit: StackFit.expand,
          children: [
            if (_error != null)
              Center(
                child: Padding(
                  padding: const EdgeInsets.all(28),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(Icons.no_photography_outlined, size: 48),
                      const SizedBox(height: 16),
                      Text(
                        'Camera unavailable\n\n${_cameraErrorMessage(_error!)}',
                        textAlign: TextAlign.center,
                      ),
                      const SizedBox(height: 18),
                      FilledButton.tonal(
                        onPressed: _retry,
                        child: const Text('Try again'),
                      ),
                    ],
                  ),
                ),
              )
            else if (camera == null || !camera.value.isInitialized)
              const Center(child: CircularProgressIndicator())
            else
              Center(child: CameraPreview(camera)),
            Container(
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [Colors.black54, Colors.transparent, Colors.black87],
                  stops: [0, .45, 1],
                ),
              ),
            ),
            SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(18),
                child: Column(
                  children: [
                    Row(
                      children: [
                        IconButton.filledTonal(
                          onPressed: _tryClose,
                          icon: const Icon(Icons.close),
                        ),
                        const Spacer(),
                        SegmentedButton<bool>(
                          segments: const [
                            ButtonSegment(value: true, label: Text('Left')),
                            ButtonSegment(value: false, label: Text('Right')),
                          ],
                          selected: {_leftSide},
                          onSelectionChanged: (value) => setState(() {
                            _leftSide = value.first;
                            // The other side's joints are elsewhere; don't
                            // let the switch complete half a rep.
                            _counter.restartCycle();
                          }),
                        ),
                      ],
                    ),
                    if (_counter.reps == 0) ...[
                      const SizedBox(height: 14),
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.black54,
                          borderRadius: BorderRadius.circular(14),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.phone_android, size: 20),
                            const SizedBox(width: 10),
                            Expanded(child: Text(widget.exercise.setupTip)),
                          ],
                        ),
                      ),
                    ],
                    const Spacer(),
                    Text(
                      '${_counter.reps}',
                      style: const TextStyle(
                        fontSize: 86,
                        fontWeight: FontWeight.w900,
                        height: .9,
                      ),
                    ),
                    const Text(
                      'REPS',
                      style: TextStyle(
                        letterSpacing: 3,
                        color: Colors.white70,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 18,
                        vertical: 12,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.black54,
                        borderRadius: BorderRadius.circular(18),
                      ),
                      child: Text(
                        _feedback,
                        style: const TextStyle(
                          fontSize: 17,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '${_angle?.toStringAsFixed(0) ?? '--'}°  •  ${_seconds}s',
                      style: const TextStyle(color: Colors.white70),
                    ),
                    const SizedBox(height: 20),
                    Row(
                      children: [
                        Expanded(
                          child: OutlinedButton(
                            onPressed: _reset,
                            child: const Text('Reset'),
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: FilledButton(
                            onPressed: _finish,
                            child: const Text('Finish workout'),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

extension _FirstOrNull<T> on Iterable<T> {
  T? get firstOrNull => isEmpty ? null : first;
}
