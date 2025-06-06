import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, jsonify

app = Flask(__name__)

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Define Rock, Paper, Scissors logic
def get_gesture(hand_landmarks):
    # Simplified logic: Count the number of fingers extended
    finger_tips = [8, 12, 16, 20]  # Indices for fingertip landmarks
    extended_fingers = 0

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            extended_fingers += 1
    
    # Mapping gestures to Rock, Paper, Scissors
    if extended_fingers == 5:
        return 'rock'
    elif extended_fingers == 3:
        return 'paper'
    elif extended_fingers == 1:
        return 'scissors'
    else:
        return 'unknown'

# Flask route for the game
@app.route('/')
def index():
    return render_template('index.html')

# Video capture route
@app.route('/video_feed')
def video_feed():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the frame with MediaPipe hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                # Get the gesture
                gesture = get_gesture(landmarks)
                # Show gesture on the frame
                cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame back to BGR and send it as the response
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
