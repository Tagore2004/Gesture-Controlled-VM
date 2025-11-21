import os
import numpy as np

DATA_DIR = "gesture_data"

# =============================
# Augmentation Functions
# =============================

def add_noise(vec, noise_level=0.005):
    return vec + np.random.normal(0, noise_level, vec.shape)

def scale(vec, s_min=0.95, s_max=1.05):
    s = np.random.uniform(s_min, s_max)
    return vec * s

def jitter(vec, jitter_strength=0.01):
    return vec + (np.random.rand(*vec.shape) - 0.5) * jitter_strength

def stretch(vec, axis_ratio=0.05):
    vec2 = vec.copy()
    # randomly stretch X (even indices) or Y (odd indices)
    if np.random.rand() > 0.5:
        factor = np.random.uniform(1 - axis_ratio, 1 + axis_ratio)
        vec2[0::2] *= factor
    else:
        factor = np.random.uniform(1 - axis_ratio, 1 + axis_ratio)
        vec2[1::2] *= factor

    return vec2

def mirror(vec):
    vec2 = vec.copy()
    # invert x-axis (all even indices)
    vec2[0::2] *= -1
    return vec2


# =============================
# Load, Augment, Save
# =============================

print("========== DATASET AUGMENTATION ==========")

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".npz")]

if not files:
    print("⚠ No dataset found in gesture_data/. Run data collection first.")
    exit()

AUG_PER_SAMPLE = 5   # each sample generates 5 new samples

for file in files:
    path = os.path.join(DATA_DIR, file)
    print(f"Processing: {path}")

    data = np.load(path)['X']
    augmented = []

    for vec in data:
        for _ in range(AUG_PER_SAMPLE):
            v = vec.copy()
            v = add_noise(v)
            v = jitter(v)
            v = scale(v)
            v = stretch(v)
            v = mirror(v)
            augmented.append(v)

    augmented = np.array(augmented, dtype=np.float32)

    # combine original + augmented
    combined = np.vstack([data, augmented])

    # overwrite old file for training
    np.savez_compressed(path, X=combined)

    print(f"✓ Augmented {file}: {len(data)} → {combined.shape[0]} samples")

print("\nAll gestures augmented successfully!")
