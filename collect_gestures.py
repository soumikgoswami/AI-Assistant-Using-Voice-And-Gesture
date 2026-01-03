# collect_gestures.py
"""
Gesture Dataset Collector using MediaPipe
Press keys to label gestures and capture data to CSV.
Example keys: f=FORWARD, b=BACK, l=LEFT, r=RIGHT, s=STOP
"""
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import os

# ----- Configuration -----
SAVE_FILE = "gesture_dataset.csv"
# Add more keys for additional gestures (circle, speed up/down, start, help)
GESTURE_KEYS = {
    'f': "FORWARD",
    'b': "BACK",
    'l': "LEFT",
    'r': "RIGHT",
    's': "STOP",
    'c': "CIRCLE",
    'u': "SPEED_UP",   # up swipe
    'd': "SLOW_DOWN",  # down swipe
    't': "START",      # thumb / poke
    'h': "HELP"
}
SAMPLES_PER_GESTURE = 400  # increase frames per gesture for better training

# ----- MediaPipe Setup -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def collect_landmarks(landmarks):
    """Flatten 21 (x, y, z) coordinates into a list"""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()


def engineered_features(landmarks):
    """Compute simple engineered features from 21x3 landmarks.
    Returns a 1D numpy array of features to augment raw landmarks.
    Features included:
      - distances: thumb_tip-index_tip, index_tip-middle_tip, palm_width (wrist->index_mcp)
      - angles: angle at index_mcp (between index_mcp->index_pip and index_mcp->wrist)
    """
    lm = np.array([[p.x, p.y, p.z] for p in landmarks])
    def dist(a, b):
        return np.linalg.norm(lm[a, :2] - lm[b, :2])

    # indices
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    INDEX_MCP = 5
    WRIST = 0

    d_thumb_index = dist(THUMB_TIP, INDEX_TIP)
    d_index_middle = dist(INDEX_TIP, MIDDLE_TIP)
    palm_size = dist(WRIST, INDEX_MCP)

    # simple angle at index_mcp using three points: index_pip(6) - index_mcp(5) - wrist(0)
    a = lm[6, :2] - lm[5, :2]
    b = lm[0, :2] - lm[5, :2]
    # angle between a and b
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    cosang = np.dot(a, b) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angle_index_mcp = np.arccos(cosang)

    return np.array([d_thumb_index, d_index_middle, palm_size, angle_index_mcp])

def main():
    cap = cv2.VideoCapture(0)
    collected_data = []

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        print("Press one of these keys to collect samples:")
        for k, v in GESTURE_KEYS.items():
            print(f"'{k}' = {v}")
        print("Press 'q' to quit and save data.\n")

        current_label = None
        counter = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                # results.multi_hand_landmarks and multi_handedness correspond by index
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if current_label:
                        # mirror left-hand landmarks so model sees consistent orientation
                        hand_label = handedness.classification[0].label if results.multi_handedness else 'Right'
                        pts = []
                        for p in hand_landmarks.landmark:
                            x = (1.0 - p.x) if hand_label.lower().startswith('left') else p.x
                            pts.append([x, p.y, p.z])
                        row = np.array(pts).flatten()
                        feats = engineered_features(pts)
                        collected_data.append([current_label] + row.tolist() + feats.tolist())
                        counter += 1
                        cv2.putText(frame, f"Recording {current_label} ({counter}/{SAMPLES_PER_GESTURE})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                        if counter >= SAMPLES_PER_GESTURE:
                            current_label = None
                            counter = 0
                            print("✅ Completed one gesture recording.")
            else:
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow("Gesture Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif chr(key) in GESTURE_KEYS.keys() and not current_label:
                current_label = GESTURE_KEYS[chr(key)]
                counter = 0
                print(f"➡ Started collecting for {current_label}")

    cap.release()
    cv2.destroyAllWindows()

    # ----- Save data -----
    if collected_data:
        df = pd.DataFrame(collected_data)
        raw_cols = [f"{coord}{i}" for i in range(21) for coord in ["x", "y", "z"]]
        eng_cols = ["d_thumb_index", "d_index_middle", "palm_size", "angle_index_mcp"]
        cols = ["gesture"] + raw_cols + eng_cols
        df.columns = cols

        if os.path.exists(SAVE_FILE):
            old = pd.read_csv(SAVE_FILE)
            df = pd.concat([old, df], ignore_index=True)

        df.to_csv(SAVE_FILE, index=False)
        print(f"💾 Saved {len(df)} total samples to {SAVE_FILE}")
    else:
        print("⚠️ No data collected.")

if __name__ == "__main__":
    main()
