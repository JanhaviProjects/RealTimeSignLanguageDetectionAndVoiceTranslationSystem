import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory with images
DATA_DIR = r'C:\Users\wankh\Desktop\Sem VIII\Sign to Speech\Data\Z'

data = []
labels = []

# Assuming you have a predefined list of labels or some way to assign labels to the images
# For example, if you have 5 classes, you can assign labels like so:
class_labels = ["WON", "HOPES", "THANKS", "CALL", "OK", "NO", "YOU", "YES", "NICE", "HELLO"]
image_index = 0  # Index to loop over class_labels

# Process each image in the directory
for img_path in os.listdir(DATA_DIR):
    img_full_path = os.path.join(DATA_DIR, img_path)
    
    # Check if the file is an image (add more formats as needed)
    if not img_full_path.endswith(('jpg', 'png', 'jpeg')):  
        continue
    
    try:
        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Failed to load {img_full_path}")
            continue  # Skip if the image couldn't be loaded

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image with MediaPipe
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []
            
            # Extract landmarks from each detected hand
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize coordinates and store them in data_aux
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Add the processed data and label
            data.append(data_aux)
            
            # Assign label based on the current index or image name
            labels.append(class_labels[image_index % len(class_labels)])  # Cycle through labels if more images

            # Optionally, print info to verify the process
            print(f"Processed {img_full_path}, Label: {class_labels[image_index % len(class_labels)]}")

            # Increment the image index
            image_index += 1

    except Exception as e:
        print(f"Error processing {img_full_path}: {e}")
        continue

# Save data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Data saved successfully with {len(data)} samples.")
