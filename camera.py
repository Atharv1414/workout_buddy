import cv2

# Use AVFoundation backend (better for macOS M1/M2)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("⚠️ Unable to access the camera. Please enable camera permissions for Terminal in System Settings → Privacy & Security → Camera.")
else:
    print("✅ Camera opened successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame not received.")
        break

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
