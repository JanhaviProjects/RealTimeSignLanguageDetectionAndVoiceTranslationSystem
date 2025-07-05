import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/wankh/Desktop/Sem VIII/Sign to Speech/Model/keras_model.h5",
                        "C:/Users/wankh/Desktop/Sem VIII/Sign to Speech/Model/labels.txt")

offset = 20
imgSize = 300
counter = 0

# labels for hand gestures
labels = ["WON", "HOPES", "THANKS", "CALL", "OK", "NO", "YOU", "YES", "NICE", "HELLO"]

prev_index = -1  # Store the previous prediction index
confidence_threshold = 0.9  # Set confidence threshold (90%)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the crop doesn't go out of bounds
        if x - offset >= 0 and y - offset >= 0 and x + w + offset <= img.shape[1] and y + h + offset <= img.shape[0]:
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        else:
            print("Invalid bounding box, skipping frame")
            continue  # Skip this frame if bounding box is out of bounds

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        # Get Prediction from the classifier (Now returns only prediction and index)
        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction, index)

        # Ensure the prediction index is valid and process the output
        if index is not None and 0 <= index < len(labels):
            if index != prev_index:
                engine.setProperty('rate', 150)  # Adjust the speaking speed if needed
                engine.say(labels[index])
                engine.runAndWait()
                prev_index = index

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                          cv2.FILLED)

            # Display the predicted label on the screen
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        else:
            print("Invalid prediction index:", index)

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    else:
        print("No hands detected")

    # Display the final output
    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
