import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping

# ------------------ Configuration ------------------
DATA_PATH = Path("video/data")  # Path where .npy files are stored
ACTIONS = np.array(['hello', 'how', 'you'])  # List of action labels
SEQUENCE_LENGTH = 30

tb_callback = TensorBoard(log_dir='video/logs')  # TensorBoard callback for visualization

# ------------------ Load Data ------------------
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

# ------------------ Define Model ------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(ACTIONS), activation='softmax'))  # Output layer

# We could also send each possible sentence generation to the LLM to generate a more correct response

# ------------------ Compile and Train ------------------
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2000, callbacks=[tb_callback, early_stop])

# ------------------ Save Model ------------------
model.save("video/models/LSTM(2).h5")  # Save the trained model to disk
