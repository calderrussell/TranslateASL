import os
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


# --- IMPORTANT: Update these variables for each model/test set you use ---
# 1. DATA_PATH: Path to your test data (.npy files), must match your dataset location.
# 2. ACTIONS: List of action labels/classes, must match what your model was trained on.
# 3. SEQUENCE_LENGTH: Number of frames per sample, must match your model's expected input.
# 4. MODEL_PATH: Path to your trained model file (.h5), must match the model you want

DATA_PATH = Path("video/data")  # Path where .npy files are stored
ACTIONS = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])  # List of action labels
SEQUENCE_LENGTH = 10

# Load the trained model
MODEL_PATH = 'video/models/LSTM_alpha.h5'
model = load_model(MODEL_PATH)

model.summary()  # Print model summary for verification

# Create test data
sequences, labels = [], []
label_map = {label: num for num, label in enumerate(ACTIONS)}


# Load .npy files and assign labels
for action in ACTIONS:
    for sequence in os.listdir(DATA_PATH / action):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            frame_path = DATA_PATH / action / sequence / f"{frame_num}.npy"
            window.append(np.load(frame_path))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)  # Shape: (num_samples, 30, 1662)
y = to_categorical(labels).astype(int)  # Convert labels to one-hot encoding

# ------------------ Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

def evaluate_model(model, X_test, y_test):
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    mcm = multilabel_confusion_matrix(ytrue, yhat)
    acc = accuracy_score(ytrue, yhat)
    return mcm, acc

# ------------------ Evaluate Model ------------------
# This function can be used to evaluate the model's performance on the test set.
print(evaluate_model(model, X_test, y_test))


# Example usage:
# mcm, acc = evaluate_model(model, X_test, y_test)
# print("Confusion Matrix:", mcm)
# print("Accuracy:", acc)

