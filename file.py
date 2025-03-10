import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)  # Allow detection of only one hand
imgSize = 400  # Increased canvas size for better fitting
counter = 0
folder = "Data/Z"

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    hands, img = detector.findHands(img, draw=False)  # Detect hand without drawing
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Create a white background

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 landmark points [(x, y, z), ...]
        x_min, y_min = np.min(lmList, axis=0)[:2]  # Get min X, Y
        x_max, y_max = np.max(lmList, axis=0)[:2]  # Get max X, Y

        # Calculate width and height of the bounding box
        w, h = x_max - x_min, y_max - y_min
        scale = (imgSize * 0.8) / max(w, h)  # Scale based on the largest dimension (80% of canvas)

        lmList = np.array(lmList)[:, :2]  # Remove Z coordinates
        lmList = (lmList - [x_min, y_min]) * scale  # Normalize and scale
        lmList += (imgSize - np.array([w, h]) * scale) / 2  # Center the landmarks

        # Draw landmarks with larger points
        for x, y in lmList.astype(int):
            cv2.circle(imgWhite, (x, y), 8, (0, 0, 0), -1)  # Increased circle size

        # Connect landmarks to form the hand structure
        connections = [[0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
                       [0, 5], [5, 6], [6, 7], [7, 8],  # Index
                       [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
                       [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
                       [0, 17], [17, 18], [18, 19], [19, 20]]  # Pinky

        for p1, p2 in connections:
            cv2.line(imgWhite, tuple(lmList[p1].astype(int)), tuple(lmList[p2].astype(int)), (0, 0, 0), 4)  # Thicker lines

        # Display the white image with landmarks
        cv2.imshow('Landmarks on White', imgWhite)

    # Show the original image with detections
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    if key == ord("s") and hands:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved image {counter}")
    elif key == ord("q"):  # Press 'q' to quit
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
