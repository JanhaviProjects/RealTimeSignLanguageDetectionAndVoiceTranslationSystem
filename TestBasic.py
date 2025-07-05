import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import mediapipe as mp
import numpy as np
import math
import pyttsx3
from collections import deque

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "C:/Users/wankh/Desktop/Sem VIII/Sign to Speech/Model/keras_model.h5",
    "C:/Users/wankh/Desktop/Sem VIII/Sign to Speech/Model/labels.txt"
)

# Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speed
tts_engine.setProperty('volume', 1.0)  # Max volume

# Constants
offset = 20
imgSize = 300
confidence_threshold = 0.8  # Minimum confidence to accept a prediction
labels = ["WON", "HOPES", "THANKS", "CALL", "OK", "NO", "YOU", "YES", "NICE", "HELLO"]  # Ensure these labels match those used in Model.py

# Moving average filter for stable predictions
prev_predictions = deque(maxlen=5)  # Stores last 5 predictions for smoothing
prev_label = None  # To prevent repeating speech

while True:
    success, img = cap.read()
    if not success:
        continue  # Skip if frame not captured

    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)  # Detect hands

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a blank white image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]

        if imgCrop.shape[0] == 0 or imgCrop.shape[1] == 0:
            continue  # Skip if crop is invalid

        aspectRatio = h / w

        # Resize maintaining aspect ratio
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Get model prediction
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        max_confidence = max(prediction)  # Get the highest confidence value

        # Print for debugging
        print(f"Raw Prediction: {prediction}, Index: {index}, Confidence: {max_confidence}")

        # Use prediction only if confidence is above threshold
        if max_confidence > confidence_threshold:
            prev_predictions.append(labels[index])

        # Smooth prediction using moving average
        if len(prev_predictions) > 0:
            most_common_label = max(set(prev_predictions), key=prev_predictions.count)
        else:
            most_common_label = labels[index]

        # Display prediction
        cv2.rectangle(imgOutput, (x-offset, y-offset-70), (x-offset+400, y-offset-50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, most_common_label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Speak the detected word (avoiding repetition)
        if most_common_label != prev_label:
            tts_engine.say(most_common_label)
            tts_engine.runAndWait()
            prev_label = most_common_label

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
