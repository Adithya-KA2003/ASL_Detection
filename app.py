from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import time
import keyboard  # Capturing key presses

app = Flask(__name__)

# ✅ Load the trained ASL model
model_path = "C:/Users/adith/OneDrive/Desktop/ASL/my_model.h5"
model = load_model(model_path, compile=False)

# ✅ Define class labels (A-Z)
class_labels = {i: chr(65 + i) for i in range(26)}

# ✅ Initialize hand detector
detector = HandDetector(maxHands=1)
imgSize = 400

# ✅ Sentence formation variables
sentence = ""
current_word = ""
last_predicted_letter = None
last_detection_time = time.time()
cooldown_period = 2.5
confirmation_threshold = 7
confirmation_counter = 0


def generate_frames():
    """Real-time ASL detection and streaming"""
    global sentence, current_word, last_predicted_letter, last_detection_time, confirmation_counter

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror image for natural interaction
        hands, frame = detector.findHands(frame, draw=False)  # Detect hands, no drawing

        # ✅ Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        predicted_letter = ""

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]

            if len(lmList) == 0:
                continue

            x_min, y_min = np.min(lmList, axis=0)[:2]
            x_max, y_max = np.max(lmList, axis=0)[:2]

            w, h = x_max - x_min, y_max - y_min
            scale = (imgSize * 0.8) / max(w, h)
            lmList = np.array(lmList)[:, :2]
            lmList = (lmList - [x_min, y_min]) * scale
            lmList += (imgSize - np.array([w, h]) * scale) / 2

            # ✅ Draw landmarks on white background
            for x, y in lmList.astype(int):
                cv2.circle(imgWhite, (x, y), 8, (0, 0, 0), -1)  # Black dots

            # ✅ Connect landmarks
            connections = [[0, 1], [1, 2], [2, 3], [3, 4],
                           [0, 5], [5, 6], [6, 7], [7, 8],
                           [0, 9], [9, 10], [10, 11], [11, 12],
                           [0, 13], [13, 14], [14, 15], [15, 16],
                           [0, 17], [17, 18], [18, 19], [19, 20]]

            for p1, p2 in connections:
                cv2.line(imgWhite, tuple(lmList[p1].astype(int)), tuple(lmList[p2].astype(int)), (0, 0, 0), 4)

            # ✅ Convert to grayscale & preprocess for model
            imgWhite_gray = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2GRAY)
            img_input = cv2.resize(imgWhite_gray, (128, 128))
            img_input = img_input / 255.0
            img_input = np.expand_dims(img_input, axis=-1)
            img_input = np.expand_dims(img_input, axis=0)

            try:
                # ✅ Model Prediction
                prediction = model.predict(img_input)
                predicted_label = np.argmax(prediction)
                predicted_letter = class_labels.get(predicted_label, "")

                # ✅ Confirmation mechanism
                if predicted_letter == last_predicted_letter:
                    confirmation_counter += 1
                else:
                    confirmation_counter = 0

                if confirmation_counter >= confirmation_threshold:
                    current_time = time.time()
                    if current_time - last_detection_time > cooldown_period:
                        current_word += predicted_letter
                        last_detection_time = current_time
                    confirmation_counter = 0

                last_predicted_letter = predicted_letter

            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

        # ✅ Keyboard Input Handling
        if keyboard.is_pressed("space"):  # Space bar adds a word
            if current_word:
                sentence += current_word + " "
                current_word = ""

        if keyboard.is_pressed("."):  # Period adds a full stop
            if sentence:
                sentence = sentence.strip() + ". "

        if keyboard.is_pressed("backspace"):  # Backspace removes the last character
            if current_word:
                current_word = current_word[:-1]
            elif sentence:
                sentence = sentence.rstrip()[:-1] + " "

        # ✅ Display detected word & sentence
        cv2.putText(imgWhite, f"Word: {current_word}", (10, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(imgWhite, f"Sentence: {sentence}", (10, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # ✅ Encode the processed white-background frame
        ret, buffer = cv2.imencode('.jpg', imgWhite)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_sentence')
def get_sentence():
    return jsonify({"sentence": sentence})


@app.route('/reset', methods=['POST'])
def reset_sentence():
    global sentence, current_word
    sentence = ""
    current_word = ""
    return jsonify({"message": "Reset successful"})


if __name__ == "__main__":
    print("🚀 Flask is running on http://127.0.0.1:5000/")
    app.run(debug=True, host="127.0.0.1", port=5000)
