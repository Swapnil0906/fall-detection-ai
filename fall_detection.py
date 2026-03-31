import cv2
import numpy as np

from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_draw


pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)

def detect_fall(landmarks):
    # Get key body points
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

    # Calculate vertical difference
    vertical_diff = abs(shoulder.y - hip.y)

    # If body is more horizontal → possible fall
    if vertical_diff < 0.35:
        return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img_rgb)

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if detect_fall(result.pose_landmarks.landmark):
            cv2.putText(frame, "FALL DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            cv2.rectangle(frame, (40, 60), (400, 130), (0, 0, 255), 2)

            print("Fall Detected!")   # terminal output
        else:
            cv2.putText(frame, "NORMAL", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (40, 60), (300, 130), (0, 255, 0), 2)

    cv2.imshow("Fall Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break


    if cv2.getWindowProperty("Fall Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()