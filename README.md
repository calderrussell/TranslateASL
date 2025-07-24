# TranslateASL

TranslateASL is an application for translating American Sign Language (ASL) into text using computer vision and machine learning. The project features two distinct models: one for recognizing individual ASL letters and another for recognizing words and phrases from video sequences.

---

## Project Structure

- `calder/` and `kalle/`: Scripts and data for letter-based ASL recognition.
- `video/`: Scripts and models for word/phrase recognition from video.
- `dataset/`: Contains ASL alphabet image datasets.
- `models/`: Stores trained models and notebooks.

---

## Models

### 1. Letter Recognition Model

- **Purpose:** Translates static images of ASL hand signs into individual letters.
- **Implementation:** Uses MediaPipe Hands for landmark extraction and an XGBoost classifier for prediction.
- **Training Data:** Images from the ASL Alphabet dataset (`dataset/asl_alphabet_train/asl_alphabet_train/`).
- **Training Pipeline:**
  - Extract hand landmarks from images using MediaPipe.
  - Filter for specific letters (e.g., 'M', 'I', 'T').
  - Store landmark features and labels.
  - Train an XGBoost model and save with label encoder.
- **Realtime Processing:** 
  - Captures webcam frames.
  - Extracts hand landmarks in real time.
  - Predicts the letter using the trained model.
  - Accumulates predicted letters into text.

### 2. Word/Phrase Recognition Model (Video LSTM)

- **Purpose:** Translates ASL gestures from video sequences into words or phrases.
- **Implementation:** Uses MediaPipe Holistic for extracting pose, face, and hand landmarks; sequences are fed into an LSTM neural network.
- **Training Data:** Video sequences collected via webcam, stored as `.npy` files in `video/data/`.
- **Training Pipeline:**
  - Collect video sequences for each action (word/phrase).
  - For each frame, extract holistic keypoints and save as numpy arrays.
  - Organize data into sequences of frames per action.
  - Train an LSTM model to classify sequences.
  - Save trained model as `.h5`.
- **Realtime Processing:** 
  - Captures live video.
  - Extracts keypoints for each frame.
  - Maintains a buffer of recent frames.
  - Predicts the word/phrase using the trained LSTM model.

---

## Data Collection Methods

- **Letter Model:** 
  - Static images processed with MediaPipe Hands.
  - Only selected letters included for training.
  - Data stored as pickled feature arrays.

- **Word Model:** 
  - Video sequences collected for each action.
  - Each frame processed with MediaPipe Holistic.
  - Keypoints saved as `.npy` files for each sequence.

---

## Training Methods

- **Letter Model:** 
  - Features: Hand landmarks.
  - Classifier: XGBoost.
  - Evaluation: Accuracy on test split.

- **Word Model:** 
  - Features: Sequence of holistic keypoints.
  - Classifier: LSTM neural network.
  - Evaluation: Categorical accuracy, early stopping, TensorBoard visualization.

---

## Realtime Processing

- **Letter Model:** 
  - Webcam feed processed frame-by-frame.
  - Hand landmarks extracted and classified.
  - Text output updated live.

- **Word Model:** 
  - Webcam feed processed as sequences.
  - Keypoints extracted for each frame.
  - Buffer of frames used for LSTM prediction.
  - Predicted word/phrase displayed live.

---

## Getting Started

1. **Install dependencies:**  
   See `calder/requirements.txt`.

2. **Prepare datasets:**  
   - Place ASL alphabet images in `dataset/asl_alphabet_train/asl_alphabet_train/`.
   - Collect video data using `video/dataCreation.py`.

3. **Train models:**  
   - Letter model: Run `calder/datasetCreation.py` and train using XGBoost.
   - Word model: Run `video/TF_LSTM.py` to train LSTM.

4. **Run realtime translation:**  
   - Letters: `calder/real3d.py`
   - Words: `video/realtimeVid.py`

---

## License

See [LICENSE](LICENSE) for details.
