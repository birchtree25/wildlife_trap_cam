### DOES NOT WORK, SAVED FOR SELF REFERENCE
### DO NOT USE





# print("Hello World")
import cv2
import torch
import PIL
from PIL import Image
import torchvision.transforms as transforms
import os 
 
# Open the video file
# vid = cv2.VideoCapture('bobcat_natgeo.mp4')
# vid = cv2.VideoCapture('outoftouch.mp4')

# # Check if video opened successfully
# if not vid.isOpened():
#     print("Error opening video file 'outoftouch.mp4'.")
#     exit()

# # Display the video frame-by-frame
# while True:
#     # Capture the next frame
#     ret, frame = vid.read()

#     # Check if frame capture was successful
#     if not ret:
#         print("No more frames to read.")
#         break

#     # Display the frame
#     cv2.imshow('Video Player', frame)

#     # Exit on 'q' key press
#     if cv2.waitKey(35) & 0xFF == ord('q'):
#         break

# # Release video capture object and destroy windows
# vid.release()
# cv2.destroyAllWindows()

# print("Video playback complete.")

# def preprocess_image(image_path):
#     img = PIL.Image.open(image_path)  # Assuming PIL for image loading
#     transform = transforms.Compose([
        # transforms.Resize(256),  # Adjust size if needed
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
#     ])
#     return transform(img).unsqueeze(0)  # Add batch dimension

# _, predicted_class_index = torch.max(prediction.data, 1)

# # Map the predicted class index to the actual class label
# class_labels = os.listdir('oregon_wildlife')  # Replace with your class labels from training
# predicted_class = class_labels[predicted_class_index.item()]

# print(f'Predicted class: {predicted_class}')
