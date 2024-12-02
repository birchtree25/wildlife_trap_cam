
import cv2
import requests
import os
import random
import string
import uuid

cv2.setUseOptimized(False)  


save_key = ord('s') #take snapshot of teh current frame using 's'


#if using a video on device
# cap = cv2.VideoCapture('greyfox.mp4')

#using device camera
cap = cv2.VideoCapture(0)

#uncomment if using ip camera ONLY
# address = "http://192.168.254.9:8080/video"
# cap.open(address)
if not cap.isOpened():
    print("Error opening video!")
    exit()

cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("No more frames in the video!")
        break

    # Display the resulting frame
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    # Save the frame if the 's' key is pressed
    if key == save_key:
        filename = "preprocessed.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")

    # Exit if 'q' key is pressed
    if key & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

print("Video playback complete.")



# Send to server
url = 'http://<server ip address>/upload' 

image_path = 'preprocessed.jpg'

with open(image_path, 'rb') as img:
    files = {'image': img}
    response = requests.post(url, files=files)

print(response.json())

