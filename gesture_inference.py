import joblib
import numpy as np
import mediapipe as mp
import cv2

# Load trained pipeline (StandardScaler + classifier)
model = joblib.load("gesture_model.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_landmarks(lm):
    # flatten 21 x,y,z into 1 x 63 vector
    return np.array([[p.x, p.y, p.z] for p in lm]).flatten().reshape(1, -1)


def main():
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for lm in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                    x = get_landmarks(lm.landmark)
                    try:
                        gesture = model.predict(x)[0]
                    except Exception:
                        gesture = "?"
                    cv2.putText(frame, f"{gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Trained Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
