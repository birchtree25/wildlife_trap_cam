
### ENSURE model_scratch.pt file is ON THE SERVER MACHINE ALONG WITH THIS CODE
### WILL CAUSE ERRORS IF NOT PRESENT
### MAKE SURE YOU HAVE CREATED FOLDERS target_animals, unsure_animals, and uploads in the same directory this file exists

import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from PIL import Image 
import torchvision.models as models
import os

import time
import shutil

from flask import Flask
from flask import jsonify
from flask import request

import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 224 x 224 x 3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 122 x 122 x 16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 56 x 56 x 32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # convolutional layer (sees 28 x 28 x 64 tensor)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        # convolutional layer (sees 14 x 14 x 128 tensor)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # dropout layer (p=0.25)
        self.dropout = nn.Dropout(0.25)

        self.conv_bn1 = nn.BatchNorm2d(224,3)
        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)
        self.conv_bn5 = nn.BatchNorm2d(128)
        self.conv_bn6 = nn.BatchNorm2d(256)

        # linear layer (64 * 4 * 4 -> 133)
        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        # linear layer (133 -> 133)
        self.fc2 = nn.Linear(512, 20)
        

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.conv_bn2(self.pool(F.relu(self.conv1(x))))
        x = self.conv_bn3(self.pool(F.relu(self.conv2(x))))
        x = self.conv_bn4(self.pool(F.relu(self.conv3(x))))
        x = self.conv_bn5(self.pool(F.relu(self.conv4(x))))
        x = self.conv_bn6(self.pool(F.relu(self.conv5(x))))
        # flatten image input
        x = x.view(-1, 256 * 7 * 7)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x


# instantiate the CNN
model_scratch = Net()
print(model_scratch)

model = Net()
model.load_state_dict(torch.load('model_scratch.pt'))
model.eval()


def preprocess_image(image_path):
    img = PIL.Image.open(image_path)  #  PIL for image loading
    transform = transforms.Compose([
        transforms.Resize(256),  # Adjust size
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization
    ])
    return transform(img).unsqueeze(0)


classes  = ['bald_eagle',
 'black_bear',
 'bobcat',
 'canada_lynx',
 'columbian_black-tailed_deer',
 'cougar',
 'coyote',
 'deer',
 'elk',
 'gray_fox',
 'gray_wolf',
 'mountain_beaver',
 'nutria',
 'raccoon',
 'raven',
 'red_fox',
 'ringtail',
 'seals',
 'sea_lions',
 'virginia_opossum']

target_class = [
    'cougar',
    'deer',
    'elk',
    'gray_fox',
    'gray_wolf',
    ]


#DEFINE TARGET FOLDERS
UPLOAD_FOLDER = "uploads"   #endpoint for server image inputs
TARGET_FOLDER = "target_animals/"  #target animals with great confidence(logit values) will go in this folder
UNSURE_TARGETS = "unsure_animals/" #Targets with less confidence will go in this folder 


def predict_and_save(image_path):
    with torch.no_grad():
        image = preprocess_image(image_path)
        prediction = model(image)
        _, predicted_class_index = torch.max(prediction.data, 1)
        predicted_class = classes[predicted_class_index.item()]
        confidence_score = torch.max(prediction.data).item()

        if confidence_score > 7.0:
            if predicted_class in target_class:
                # Get creation/modification time
                creation_time = os.path.getctime(image_path)  # Use getmtime() if getctime() is unavailable
                formatted_date = time.strftime('%Y%m%d_%H%M%S', time.localtime(creation_time))

                # Rename the file with the date and class label
                new_filename = f"{formatted_date}_{predicted_class}.jpg"
                target_path = os.path.join(TARGET_FOLDER, new_filename)
                
                # Move the file to the target folder
                shutil.move(image_path, target_path)
                print(f"Tensor Value: {confidence_score}")
                print(f"Image moved to: {target_path}")
                return {
                    "status": "success",
                    "message": f"Image classified as {predicted_class} with confidence {confidence_score:.2f} and saved to {target_path}"
                }
            else:
                print(f"Predicted class: {predicted_class} is not in target_class.")
                return {
                    "status": "ignored",
                    "message": f"Image classified as {predicted_class} with confidence {confidence_score:.2f} but not moved (not in target_class)"
                }
            
        elif confidence_score > 4.0 and confidence_score <=7.0:
            if predicted_class in target_class:
                # Get creation/modification time
                creation_time = os.path.getctime(image_path)  # Use getmtime() if getctime() is unavailable
                formatted_date = time.strftime('%Y%m%d_%H%M%S', time.localtime(creation_time))

                # Rename the file with the date and class label
                new_filename = f"{formatted_date}_{predicted_class}.jpg"
                target_path = os.path.join(UNSURE_TARGETS, new_filename)
                
                # Move the file to the target folder
                shutil.move(image_path, target_path)
                print(f"Tensor Value: {confidence_score}")
                print(f"Image moved to: {target_path}")
                return {
                    "status": "success",
                    "message": f"Image classified as {predicted_class} with confidence {confidence_score:.2f} and saved to {target_path}"
                }
            


        else:
            print(f"Predicted Class: {predicted_class}")
            print("Prediction confidence too low. Classification: NULL.")
            return {
                "status": "low_confidence",
                "message": "Image classification confidence too low."
            }


app = Flask(__name__)

# Directory to save uploaded images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    image = request.files['image']

    if image.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400

    # Save the image
    file_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(file_path)

    try:
        predict_and_save(file_path)
    except Exception as exc:
        return jsonify({"Error:" f"Prediction Failed: {str(exc)}"}), 500
    

    return jsonify({"message": "Image uploaded and Predicted successfully", "file_path": file_path}), 200



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)




























































































