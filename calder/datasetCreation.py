import os
import pickle
import random

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './dataset/asl_alphabet_train/asl_alphabet_train/'

EXCLUDE_LABELS = {"del", "nothing", "space"}
INCLUDE_LABELS = {'M', 'I', 'T'}

MAX_IMAGES_PER_CLASS = 500  # Change to None to use full set


data = []
labels = []

# Progress bar
all_dirs = [
    d for d in os.listdir(DATA_DIR)
    if not d.startswith('.') and d  in INCLUDE_LABELS
]
for dir_ in tqdm(all_dirs, desc="Processing classes"):
    dir_path = os.path.join(DATA_DIR, dir_)
    image_files = [f for f in os.listdir(dir_path) if not f.startswith('.')]
    
    # Randomize image selection
    if MAX_IMAGES_PER_CLASS and len(image_files) > MAX_IMAGES_PER_CLASS:
        image_files = random.sample(image_files, MAX_IMAGES_PER_CLASS)
    
    used_images = 0

    for img_name in image_files:
        # No need to check MAX_IMAGES_PER_CLASS here since we've already sampled
        img_path = os.path.join(dir_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: could not read image {img_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_, y_ = [], []

            for landmark in results.multi_hand_landmarks[0].landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in results.multi_hand_landmarks[0].landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)
            used_images += 1

    print(f"[✓] Processed {used_images} images for '{dir_}'")

output_dir = os.path.join('.', 'calder/data')
os.makedirs(output_dir, exist_ok=True)

# Save to pickle in the Calder folder
pickle_path = os.path.join(output_dir, 'data.pickle')
with open(pickle_path, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n✅ Finished! Saved {len(data)} samples to {pickle_path}.")