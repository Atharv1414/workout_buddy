"""Small camera diagnostic for Workout Buddy."""

import cv2

from buddy import open_camera


def main() -> int:
    camera = open_camera(0)
    if not camera.isOpened():
        print("Unable to open camera 0. Check operating-system camera permissions.")
        return 1
    print("Camera opened. Press q or Esc to quit.")
    try:
        while True:
            received, frame = camera.read()
            if not received:
                print("No camera frame received.")
                return 1
            cv2.imshow("Workout Buddy - Camera Test", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return 0
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
