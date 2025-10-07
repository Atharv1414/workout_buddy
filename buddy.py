import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Helper function to calculate angle between joints
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Ask for workout type
workout_type = input("Enter workout type (squat / pushup / bicep curl): ").strip().lower()

# Open camera with AVFoundation backend (for macOS)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("⚠️ Unable to access the camera. Go to System Settings → Privacy & Security → Camera, and enable access for Terminal or Python.")
    exit()

print("✅ Camera started successfully! Press 'q' to quit.")

# Pose detection
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received. Try restarting the app.")
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            if workout_type == 'squat':
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle)
                cv2.putText(image, f'Knee Angle: {int(angle)}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if angle < 70:
                    feedback = "Down too low!"
                elif 70 <= angle <= 100:
                    feedback = "Perfect squat!"
                else:
                    feedback = "Go lower!"

            elif workout_type == 'pushup':
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, f'Elbow Angle: {int(angle)}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if angle < 90:
                    feedback = "Push up!"
                elif 90 <= angle <= 160:
                    feedback = "Good form!"
                else:
                    feedback = "Go lower!"

            elif workout_type == 'bicep curl':
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(image, f'Arm Angle: {int(angle)}', (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if angle > 160:
                    feedback = "Lower your arm"
                elif 40 < angle < 160:
                    feedback = "Good rep!"
                else:
                    feedback = "Curl up!"

            else:
                feedback = "Invalid workout type"

            cv2.putText(image, feedback, (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        except Exception:
            pass

        # Draw landmarks
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        cv2.imshow('Workout Buddy 🏋️‍♂️', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
