import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from joblib import dump

DATA_DIR = "gesture_data"
MODEL_PATH = "gesture_model.joblib"

print("========== IMPROVED MODEL TRAINING ==========")

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]
if not files:
    print("⚠ No gesture dataset found. Collect data first.")
    exit()

X_all = []
y_all = []
label_to_idx = {}
idx = 0

# Load all gesture files
for f in files:
    path = os.path.join(DATA_DIR, f)
    data = np.load(path)['X']

    gesture_label = f.split("_")[0]
    if gesture_label not in label_to_idx:
        label_to_idx[gesture_label] = idx
        idx += 1

    y = np.full((len(data),), label_to_idx[gesture_label])
    X_all.append(data)
    y_all.append(y)

X_all = np.vstack(X_all)
y_all = np.hstack(y_all)

# Shuffle
X_all, y_all = shuffle(X_all, y_all, random_state=42)

# Scale for better accuracy
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

# Stronger MLP model
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    max_iter=1200,
    learning_rate='adaptive',
    solver='adam',
    early_stopping=True,
    random_state=42
)

print("Training model… this may take 1–3 minutes.")
clf.fit(X_scaled, y_all)

dump({"model": clf, "scaler": scaler, "gesture_map": label_to_idx}, MODEL_PATH)

print(f"✓ Model trained successfully and saved to {MODEL_PATH}")
