import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LINE_COLOR = (255, 255, 0)  # cyan
POINT_COLOR = (0, 0, 255)     # red

MODEL_PATH = "hand_landmarker.task"  # <- tu podaj ścieżkę do pliku modelu

# Połączenia między punktami dłoni (21 landmarków)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # palm edge
]

def draw_landmarks(frame, hand_landmarks_list):
    h, w, _ = frame.shape

    for hand_landmarks in hand_landmarks_list:
        points = []
        for lm in hand_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            points.append((x, y))

        # linie
        for start_idx, end_idx in HAND_CONNECTIONS:
            x1, y1 = points[start_idx]
            x2, y2 = points[end_idx]
            cv2.line(frame, (x1, y1), (x2, y2), LINE_COLOR, 2)

        # punkty
        for x, y in points:
            cv2.circle(frame, (x, y), 4, POINT_COLOR, -1)

def main():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Download a Hand Landmarker .task model and put it next to this script."
        )

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                draw_landmarks(frame, result.hand_landmarks)

                # opcjonalnie: podpis left/right nad nadgarstkiem
                if result.handedness:
                    h, w, _ = frame.shape
                    for hand_landmarks, handedness_list in zip(result.hand_landmarks, result.handedness):
                        wrist = hand_landmarks[0]
                        x = int(wrist.x * w)
                        y = int(wrist.y * h) - 10

                        label = handedness_list[0].category_name
                        score = handedness_list[0].score
                        text = f"{label} {score:.2f}"
                        cv2.putText(
                            frame,
                            text,
                            (x, max(y, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 0, 0),
                            2,
                        )

            cv2.imshow("Hand Landmarker - Tasks API", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()