import cv2
import mediapipe as mp
import numpy as np
import math

# ---------------- MediaPipe Setup ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ---------------- Webcam ----------------
cap = cv2.VideoCapture(0)

spawned_objects = []  # AR objects list

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

print("🤖 Hand AR Started | Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    show_menu = False

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            # Landmarks
            thumb = hand.landmark[4]
            index = hand.landmark[8]
            palm = hand.landmark[0]

            thumb_pos = (int(thumb.x * w), int(thumb.y * h))
            index_pos = (int(index.x * w), int(index.y * h))
            palm_pos = (int(palm.x * w), int(palm.y * h))

            # ---------------- 🤏 Pinch Detection ----------------
            pinch_dist = distance(thumb_pos, index_pos)

            if pinch_dist < 30:
                spawned_objects.append(index_pos)
                cv2.putText(frame, "OBJECT SPAWNED!",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 3)

            # ---------------- 🧊 3D Cube Following Finger ----------------
            cube_size = 40
            x, y = index_pos

            # Front face
            cv2.rectangle(frame,
                          (x - cube_size, y - cube_size),
                          (x + cube_size, y + cube_size),
                          (255, 0, 0), 3)

            # Back face (fake depth)
            offset = 20
            cv2.rectangle(frame,
                          (x - cube_size + offset, y - cube_size - offset),
                          (x + cube_size + offset, y + cube_size - offset),
                          (255, 0, 0), 3)

            # Connect faces
            cv2.line(frame, (x - cube_size, y - cube_size),
                     (x - cube_size + offset, y - cube_size - offset), (255, 0, 0), 2)
            cv2.line(frame, (x + cube_size, y - cube_size),
                     (x + cube_size + offset, y - cube_size - offset), (255, 0, 0), 2)
            cv2.line(frame, (x - cube_size, y + cube_size),
                     (x - cube_size + offset, y + cube_size - offset), (255, 0, 0), 2)
            cv2.line(frame, (x + cube_size, y + cube_size),
                     (x + cube_size + offset, y + cube_size - offset), (255, 0, 0), 2)

            # ---------------- ✋ Hand-Controlled Menu ----------------
            if pinch_dist > 80:
                show_menu = True

    # Draw spawned AR objects
    for obj in spawned_objects:
        cv2.circle(frame, obj, 12, (0, 255, 255), -1)

    # AR Menu
    if show_menu:
        cv2.rectangle(frame, (20, 80), (260, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 80), (260, 200), (0, 255, 0), 2)

        cv2.putText(frame, "AR MENU",
                    (80, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        cv2.putText(frame, "1. Spawn Object",
                    (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        cv2.putText(frame, "2. Move Cube",
                    (40, 165),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

        cv2.putText(frame, "3. Exit (q)",
                    (40, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)

    cv2.imshow("Hand Gesture AR System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
