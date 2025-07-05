import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os  # Import os for folder handling

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

# Define the folder where images will be saved
folder = r"C:\Users\wankh\Desktop\Sem VIII\Sign to Speech\Data\Z"

# Create the folder if it doesn't exist
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure cropping region is within bounds
        y1, y2 = max(y - offset, 0), min(y + h + offset, img.shape[0])
        x1, x2 = max(x - offset, 0), min(x + w + offset, img.shape[1])

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:  # Check if cropping is successful
            print("Error: Cropped image is empty!")
            continue

        aspectRatio = h / w

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

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)

    # Capture the key press only once
    key = cv2.waitKey(1)

    if key == ord("q"):  # Press 'q' to exit
        break
    elif key == ord("s"):  # Press 's' to save the image
        print("Saving image...")
        counter += 1
        img_path = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(f'C:/Users/wankh/Desktop/Sem VIII/Sign to Speech/Data/Z/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved Image {counter}: {img_path}")

cap.release()
cv2.destroyAllWindows()
