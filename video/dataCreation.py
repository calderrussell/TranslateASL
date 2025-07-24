import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm

# ------------------ Configuration ------------------
# Root directory for storing keypoint data
DATA_PATH = Path("video/data")

# List of actions (labels) to collect data for
# ACTIONS = np.array(['hello', 'how', 'you'])
ACTIONS = np.array(['z'])

# Number of sequences (videos) per action
NUM_SEQUENCES = 40

# Number of frames per sequence
SEQUENCE_LENGTH = 10

# Create directory structure for saving data
tqdm.write("Creating directories...")
for action in ACTIONS:
    for sequence in range(NUM_SEQUENCES):
        (DATA_PATH / action / str(sequence)).mkdir(parents=True, exist_ok=True)

# ------------------ MediaPipe Setup ------------------
mp_holistic = mp.solutions.holistic  # Load holistic model (pose, face, hands)
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for landmarks

# Updated face connections reference due to MediaPipe API changes
FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION

def mediapipe_detection(image, model):
    """Process an image through the MediaPipe model and return the results."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Improve performance
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    """Extract flattened arrays of keypoints from holistic model results."""
    def landmark_array(landmarks, size):
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten() if landmarks else np.zeros(size)

    pose = landmark_array(results.pose_landmarks, 33 * 3)
    face = landmark_array(results.face_landmarks, 468 * 3)
    lh = landmark_array(results.left_hand_landmarks, 21 * 3)
    rh = landmark_array(results.right_hand_landmarks, 21 * 3)
    return np.concatenate([pose, face, lh, rh])


    
def draw_landmarks(image, results):
    """Draw pose, face, and hand landmarks on the image."""
    mp_drawing.draw_landmarks(image, results.face_landmarks, FACE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

# ------------------ Data Collection ------------------
# Initialize webcam
cap = cv2.VideoCapture(0)

# Use MediaPipe holistic model with detection and tracking confidence
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in ACTIONS:
        for sequence in range(NUM_SEQUENCES):

            # Wait before starting each new video
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam. Exiting...")
                cap.release()
                cv2.destroyAllWindows()
                exit()

            # Display startup message for this sequence
            cv2.putText(frame, 'STARTING COLLECTION', (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f'Collecting frames for {action.upper()} - Video {sequence}', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(1000)  # Wait 1 second

            for frame_num in tqdm(range(SEQUENCE_LENGTH), desc=f"{action}-{sequence}"):

                # Read frame from webcam
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam. Exiting...")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                # Run detection and extract keypoints
                image, results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)

                # Draw landmarks on the image
                draw_landmarks(image, results)

                # Save keypoints as .npy file
                file_path = DATA_PATH / action / str(sequence) / f"{frame_num}.npy"
                np.save(file_path, keypoints)

                # Display progress on screen
                cv2.putText(image, f"Collecting {action.upper()} - Seq {sequence}, Frame {frame_num}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Data Collection', image)
                cv2.waitKey(10)

# Release webcam and close display windows
cap.release()
cv2.destroyAllWindows()

