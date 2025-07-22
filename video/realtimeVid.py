import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import mediapipe as mp

# ------------------ Configuration ------------------
ACTIONS = np.array(['hello', 'how are you'])  # List of action labels
SEQUENCE_LENGTH = 30  # Number of frames per prediction
MODEL_PATH = 'video/models/LSTM(how are you).h5'  # Path to trained model

# Load trained model
model = load_model(MODEL_PATH)

# ------------------ MediaPipe Setup ------------------
mp_holistic = mp.solutions.holistic  # Load holistic model (pose, face, hands)
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for landmarks
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
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

# ------------------ Real-Time Detection ------------------
cap = cv2.VideoCapture(0)  # Open webcam

sequence = deque(maxlen=SEQUENCE_LENGTH)  # Holds last N keypoint frames
threshold = 0.8  # Prediction confidence threshold

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Webcam read failed")
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        draw_landmarks(image, results)

        # Extract keypoints and append to sequence
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Only predict when we have enough frames
        if len(sequence) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, 1662)
            res = model.predict(input_data, verbose=0)[0]
            confidence = np.max(res)
            predicted_action = ACTIONS[np.argmax(res)]

            # Display prediction only if confidence is high enough
            if confidence > threshold:
                cv2.putText(image, f'{predicted_action.upper()} ({confidence:.2f})', (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show frame
        cv2.imshow('Real-Time Detection', image)

        # Quit if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
