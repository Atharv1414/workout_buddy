"""Real-time exercise form feedback and repetition counting."""

from __future__ import annotations

import argparse
import math
import platform
import time
from dataclasses import dataclass
from typing import Sequence

Point = Sequence[float]


def calculate_angle(a: Point, b: Point, c: Point) -> float:
    """Return the smaller angle ABC in degrees."""
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = abs(math.degrees(radians))
    return 360 - angle if angle > 180 else angle


@dataclass(frozen=True)
class ExerciseConfig:
    label: str
    joint: str
    down_angle: float
    up_angle: float
    down_feedback: str
    up_feedback: str


EXERCISES = {
    "squat": ExerciseConfig("Squat", "knee", 95, 160, "Drive up", "Lower with control"),
    "pushup": ExerciseConfig("Push-up", "elbow", 90, 155, "Push up", "Lower with control"),
    "curl": ExerciseConfig("Bicep curl", "elbow", 55, 150, "Lower with control", "Curl up"),
}
ALIASES = {"push-up": "pushup", "push up": "pushup", "bicep curl": "curl", "bicep-curl": "curl"}


@dataclass
class RepCounter:
    """Hysteresis-based counter that avoids duplicate reps around a threshold."""

    exercise: str
    reps: int = 0
    stage: str = "ready"

    def update(self, angle: float) -> str:
        config = EXERCISES[self.exercise]
        if self.exercise in {"squat", "pushup"}:
            if angle <= config.down_angle:
                self.stage = "down"
            elif angle >= config.up_angle:
                if self.stage == "down":
                    self.reps += 1
                self.stage = "up"
            return config.down_feedback if self.stage == "down" else config.up_feedback

        if angle >= config.up_angle:
            self.stage = "down"
        elif angle <= config.down_angle:
            if self.stage == "down":
                self.reps += 1
            self.stage = "up"
        return config.down_feedback if self.stage == "up" else config.up_feedback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", nargs="?", choices=EXERCISES, help="exercise to track")
    parser.add_argument("--camera", type=int, default=0, help="camera device index (default: 0)")
    parser.add_argument("--side", choices=("left", "right"), default="left", help="body side to track")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror the camera preview")
    return parser.parse_args()


def choose_exercise(value: str | None) -> str:
    if value:
        return value
    entered = input("Exercise (squat / pushup / curl): ").strip().lower()
    entered = ALIASES.get(entered, entered)
    if entered not in EXERCISES:
        raise ValueError(f"Unknown exercise: {entered or '(empty)'}")
    return entered


def open_camera(index: int):
    import cv2

    backend = cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera


def landmark_point(landmarks, pose, side: str, joint: str) -> tuple[float, float, float]:
    landmark = getattr(pose.PoseLandmark, f"{side.upper()}_{joint.upper()}")
    item = landmarks[landmark.value]
    return item.x, item.y, item.visibility


def tracked_points(landmarks, pose, exercise: str, side: str):
    joints = ("hip", "knee", "ankle") if EXERCISES[exercise].joint == "knee" else (
        "shoulder", "elbow", "wrist"
    )
    return [landmark_point(landmarks, pose, side, joint) for joint in joints]


def draw_panel(image, counter: RepCounter, angle: float | None, feedback: str, fps: float) -> None:
    import cv2

    cv2.rectangle(image, (0, 0), (430, 155), (25, 25, 25), -1)
    cv2.putText(image, EXERCISES[counter.exercise].label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    cv2.putText(image, f"REPS  {counter.reps}", (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (80, 220, 120), 3)
    angle_text = "--" if angle is None else f"{angle:.0f} deg"
    cv2.putText(image, f"Angle: {angle_text}   FPS: {fps:.0f}", (18, 112), cv2.FONT_HERSHEY_SIMPLEX, .62, (220, 220, 220), 1)
    cv2.putText(image, feedback, (18, 142), cv2.FONT_HERSHEY_SIMPLEX, .66, (90, 210, 255), 2)


def run(exercise: str, camera_index: int, side: str, mirror: bool) -> int:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as error:
        print("Missing camera dependencies. Run: python -m pip install -r requirements.txt")
        print(f"Details: {error}")
        return 1

    pose_module = mp.solutions.pose
    drawing = mp.solutions.drawing_utils
    camera = open_camera(camera_index)
    if not camera.isOpened():
        print("Unable to open the camera. Check camera permissions or try --camera 1.")
        return 1

    counter = RepCounter(exercise)
    previous_time = time.perf_counter()
    fps = 0.0
    print("Camera started. Press q or Esc to quit, r to reset the rep count.")

    try:
        with pose_module.Pose(min_detection_confidence=.6, min_tracking_confidence=.6) as detector:
            while camera.isOpened():
                received, frame = camera.read()
                if not received:
                    print("No camera frame received.")
                    return 1
                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = detector.process(rgb)
                feedback = "Move fully into frame"
                angle = None

                if result.pose_landmarks:
                    points = tracked_points(result.pose_landmarks.landmark, pose_module, exercise, side)
                    if min(point[2] for point in points) >= .55:
                        angle = calculate_angle(points[0], points[1], points[2])
                        feedback = counter.update(angle)
                    else:
                        feedback = f"Keep your {side} side visible"
                    drawing.draw_landmarks(frame, result.pose_landmarks, pose_module.POSE_CONNECTIONS)

                now = time.perf_counter()
                instant_fps = 1 / max(now - previous_time, 1e-6)
                fps = instant_fps if fps == 0 else fps * .9 + instant_fps * .1
                previous_time = now
                draw_panel(frame, counter, angle, feedback, fps)
                cv2.imshow("Workout Buddy", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    counter = RepCounter(exercise)
        return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()
    try:
        exercise = choose_exercise(args.exercise)
    except ValueError as error:
        print(error)
        return 2
    return run(exercise, args.camera, args.side, not args.no_mirror)


if __name__ == "__main__":
    raise SystemExit(main())
