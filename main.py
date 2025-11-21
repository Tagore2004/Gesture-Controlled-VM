import sys
import time
import threading
import os
import math
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
from PyQt5 import QtCore, QtWidgets, QtGui
import pyautogui
from pynput.mouse import Controller as PynputMouse, Button as PynputButton
from sklearn.preprocessing import StandardScaler
from joblib import load

MODEL_PATH = "gesture_model.joblib"
DATA_DIR = "gesture_data"
os.makedirs(DATA_DIR, exist_ok=True)

def normalized_landmarks(landmarks):
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    out = []
    for lm in landmarks:
        out.append(lm.x - base_x)
        out.append(lm.y - base_y)
    return np.array(out, dtype=np.float32)


# ------------------ Gesture Classifier ------------------
class GestureClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.gesture_map = {}
        self.inverse_map = {}

    def load(self, path=MODEL_PATH):
        if os.path.exists(path):
            data = load(path)
            self.model = data.get("model", None)
            self.scaler = data.get("scaler", None)
            self.gesture_map = data.get("gesture_map", {})
            self.inverse_map = {v: k for k, v in self.gesture_map.items()}
            return True
        return False

    def predict(self, vec):
        if self.scaler:
            vec = self.scaler.transform([vec])[0]
        return self.model.predict([vec])[0]


# ------------------ Hand Detector ------------------
class HandDetector:
    def __init__(self):
        self.mp = mp.solutions.hands
        self.hands = self.mp.Hands(max_num_hands=1,
                                   min_detection_confidence=0.6,
                                   min_tracking_confidence=0.6)

    def detect(self, frame_rgb):
        results = self.hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None


# ------------------ Mouse Controller ------------------
class MouseController:
    def __init__(self):
        self.mouse = PynputMouse()
        self.dragging = False

    def move(self, x, y):
        try:
            self.mouse.position = (x, y)
        except:
            pass

    def left_click(self):
        self.mouse.click(PynputButton.left, 1)

    def right_click(self):
        self.mouse.click(PynputButton.right, 1)

    def start_drag(self):
        if not self.dragging:
            self.dragging = True
            self.mouse.press(PynputButton.left)

    def stop_drag(self):
        if self.dragging:
            self.dragging = False
            self.mouse.release(PynputButton.left)

    def scroll(self, dy):
        pyautogui.scroll(int(dy))


# ------------------ ENGINE ------------------
class GestureMouseEngine(QtCore.QObject):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.cap = cv2.VideoCapture(config["camera_index"])
        self.detector = HandDetector()
        self.classifier = GestureClassifier()
        self.classifier.load(MODEL_PATH)

        self.mouse = MouseController()
        self.running = False

        self.prev_cursor = None
        self.smoothing = config["smoothing"]
        self.sensitivity = config["sensitivity"]
        self.landmark_buffer = deque(maxlen=5)

    def run(self):
        self.running = True
        
        screen_w, screen_h = pyautogui.size()

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lms = self.detector.detect(rgb)

            label = None

            # -------- CURSOR MOVEMENT --------
            if lms:
                index_tip = lms.landmark[8]
                x = np.interp(index_tip.x, [0, 1], [0, screen_w])
                y = np.interp(index_tip.y, [0, 1], [0, screen_h])

                if self.prev_cursor is None:
                    final_x, final_y = x, y
                else:
                    final_x = self.prev_cursor[0] + (x - self.prev_cursor[0]) * (1 - self.smoothing)
                    final_y = self.prev_cursor[1] + (y - self.prev_cursor[1]) * (1 - self.smoothing)

                dx = (final_x - self.prev_cursor[0] if self.prev_cursor else 0) * self.sensitivity
                dy = (final_y - self.prev_cursor[1] if self.prev_cursor else 0) * self.sensitivity

                final_x = (self.prev_cursor[0] if self.prev_cursor else final_x) + dx
                final_y = (self.prev_cursor[1] if self.prev_cursor else final_y) + dy

                self.prev_cursor = (final_x, final_y)

                # MOVE THE CURSOR (fixed)
                self.mouse.move(final_x, final_y)

                # -------- GESTURE PREDICTION --------
                vec = normalized_landmarks(lms.landmark)
                if vec is not None:
                    self.landmark_buffer.append(vec)
                    smoothed_vec = np.mean(np.array(self.landmark_buffer), axis=0)
                    pred = self.classifier.predict(smoothed_vec)
                    label = self.classifier.inverse_map[pred]

                    self.perform_action(label, lms)

            # Draw + show label
            if lms:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, lms, mp.solutions.hands.HAND_CONNECTIONS
                )

            if label:
                cv2.putText(frame, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            self.frame_ready.emit(frame)
            time.sleep(0.01)

        self.cap.release()

    def perform_action(self, label, lms):
        action = self.config["gesture_action_map"].get(label)

        if action == "left_click":
            self.mouse.left_click()

        elif action == "right_click":
            self.mouse.right_click()

        elif action == "double_click":
            self.mouse.left_click()
            time.sleep(0.1)
            self.mouse.left_click()

        elif action == "drag_toggle":
            if not self.mouse.dragging:
                self.mouse.start_drag()
            else:
                self.mouse.stop_drag()

        elif action == "scroll":
            idx = lms.landmark[8]
            mid = lms.landmark[12]
            dy = (mid.y - idx.y) * 40
            self.mouse.scroll(dy)


# ------------------ GUI ------------------
class ControlPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Improved Gesture Mouse")
        self.resize(1100, 700)

        self.config = {
            "camera_index": 0,
            "smoothing": 0.8,
            "sensitivity": 0.6,
            "gesture_action_map": {
                "open_palm": "none",
                "pinch": "left_click",
                "fist": "right_click",
                "two_fingers": "scroll",
                "thumb_up": "drag_toggle",
                "peace": "double_click"
            }
        }

        layout = QtWidgets.QHBoxLayout(self)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)

        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right)

        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        h = QtWidgets.QHBoxLayout()
        h.addWidget(self.start_btn)
        h.addWidget(self.stop_btn)
        right.addLayout(h)

        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)

        right.addWidget(QtWidgets.QLabel("Gesture → Action Mapping"))
        self.mapping_edit = QtWidgets.QTextEdit()
        self.mapping_edit.setPlainText("\n".join([f"{g}:{a}" for g, a in self.config["gesture_action_map"].items()]))
        right.addWidget(self.mapping_edit)

        self.status = QtWidgets.QLabel("Idle")
        right.addWidget(self.status)

        self.engine = None
        self.thread = None

    def start(self):
        mapping = {}
        for line in self.mapping_edit.toPlainText().splitlines():
            if ":" in line:
                k, v = line.split(":")
                mapping[k.strip()] = v.strip()

        self.config["gesture_action_map"] = mapping

        self.engine = GestureMouseEngine(self.config)
        self.engine.frame_ready.connect(self.update_frame)

        self.thread = threading.Thread(target=self.engine.run, daemon=True)
        self.thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop(self):
        if self.engine:
            self.engine.stop()

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    @QtCore.pyqtSlot(np.ndarray)
    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, w * ch, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.video_label.size(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)


def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = ControlPanel()
    ui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
