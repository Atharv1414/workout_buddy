/// Rotation, in degrees, that ML Kit must apply to a camera frame so the
/// person appears upright.
///
/// iOS frames only need the sensor orientation. Android frames also depend on
/// how the device is currently held, and the front camera rotates the other
/// way because its image is mirrored. Mirrors the calculation in the
/// google_mlkit_commons example. Returns null for an unknown device rotation.
int? frameRotationDegrees({
  required bool isAndroid,
  required int sensorOrientation,
  required bool frontCamera,
  required int? deviceRotation,
}) {
  if (!isAndroid) return sensorOrientation;
  if (deviceRotation == null) return null;
  return frontCamera
      ? (sensorOrientation + deviceRotation) % 360
      : (sensorOrientation - deviceRotation + 360) % 360;
}
