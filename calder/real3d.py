import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter

# Load model and label encoder
model_dict = pickle.load(open('calder/models/model.p', 'rb'))
model = model_dict['model']

# Initialize prediction buffer and timer
prediction_buffer = []
accumulated_text = ""
last_collection_time = time.time()
collecting = True

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is phone, 1 is computer

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Initialize per-hand data
            data_aux = []
            x_ = []
            y_ = []
            z_ = []

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                x_.append(x)
                y_.append(y)
                z_.append(z)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                data_aux.append(z - min(z_))

            if len(data_aux) == 21*3:  # 21 landmarks * 3 coords
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]

                if collecting:
                    prediction_buffer.append(predicted_character)


                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]
                prediction_buffer.append(predicted_character)


                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

                # Draw label bar
                bar_height = 40
                cv2.rectangle(frame, (x1, y1 - bar_height), (x2, y1), (255, 255, 255), -1)
                cv2.putText(frame, predicted_character, (x1 + 10, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                

    # Store accumulated predictions
    if collecting and (current_time - last_collection_time >= 2.0):
        if prediction_buffer:
            most_common = Counter(prediction_buffer).most_common(1)[0][0]
            accumulated_text += most_common

        prediction_buffer = []
        collecting = False
        last_collection_time = current_time  # mark time to start pause

    elif not collecting and (current_time - last_collection_time >= 0.4):
        collecting = True
        last_collection_time = current_time  # restart collection cycle

    # Display accumulated result
    cv2.putText(frame, f"Final: {accumulated_text}", (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('Live ASL Translation', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):  # Handle backspace/delete
        accumulated_text = accumulated_text[:-1]
    elif key == ord('r'):  # Reset
        accumulated_text = ""

cap.release()
cv2.destroyAllWindows()
