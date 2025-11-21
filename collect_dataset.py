import cv2
import mediapipe as mp
import numpy as np
import time
import os

# -------- settings --------
GESTURES = [
    "open_palm",      # none
    "pinch",          # left_click
    "fist",           # right_click
    "two_fingers",    # scroll
    "thumb_up",       # drag_toggle
    "peace"           # double_click
]

SAMPLES_PER_GESTURE = 500    # recommended: 300–1000
DATA_DIR = "gesture_data"
os.makedirs(DATA_DIR, exist_ok=True)

# -------- MediaPipe Hands --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Normalize landmarks exactly like main.py
def normalized_landmarks(landmarks):
    if not landmarks:
        return None

    base_x = landmarks[0].x
    base_y = landmarks[0].y

    out = []
    for lm in landmarks:
        out.append(lm.x - base_x)
        out.append(lm.y - base_y)

    return np.array(out, dtype=np.float32)


# -------- data collection loop --------
cap = cv2.VideoCapture(0)

print("\n==============================")
print(" Gesture Dataset Collector")
print("==============================")
print("Instructions:")
print("• Show the gesture when prompted")
print("• Keep hand steady")
print("• Press 'q' anytime to quit")
print("==============================\n")

for gesture in GESTURES:

    print(f"\nPrepare to collect samples for: {gesture}")
    print("Starting in 3 seconds...")
    time.sleep(3)

    samples = []
    saved = 0

    while saved < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            vec = normalized_landmarks(lm.landmark)

            if vec is not None:
                samples.append(vec)
                saved += 1

        # display progress
        cv2.putText(frame,
                    f"{gesture}  {saved}/{SAMPLES_PER_GESTURE}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Collecting", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # save file
    if samples:
        out_path = os.path.join(DATA_DIR, f"{gesture}_{int(time.time())}.npz")
        np.savez_compressed(out_path, X=np.array(samples))
        print(f"✓ Saved {len(samples)} samples → {out_path}")
    else:
        print(f"⚠ No samples collected for {gesture}")

cap.release()
cv2.destroyAllWindows()

print("\n==============================")
print(" Dataset Generation Complete!")
print("==============================")
print(f"Files saved in folder: {DATA_DIR}/")

