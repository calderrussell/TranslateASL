import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('kalle/xgboost.p', 'rb'))
model = model_dict['model']
label_encoder = model_dict['label_encoder']

cap = cv2.VideoCapture(1)  # 0 is phone, 1 is computer

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        print(f"Feature vector length: {len(data_aux)}")
        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = label_encoder.inverse_transform(prediction)[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        
        # Draw a solid white bar above the rectangle
        bar_height = 40  # You can adjust the height as needed
        cv2.rectangle(frame, (x1, y1 - bar_height), (x2, y1), (255, 255, 255), -1)
        
        # Draw the predicted character on the white bar
        cv2.putText(frame, predicted_character, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()