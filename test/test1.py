###IMAGES ONLY###


import numpy as np
import PIL
import torch
import torchvision.transforms as transforms
from PIL import Image 
import torchvision.models as models
import os
import os

#from os import scandir

# class_labels = []
# for entry in scandir('oregon_wildlife'):
#     if not entry.is_dir(): 
#         class_labels.append(entry.name)

# class_labels.sort()


import torch.nn as nn
import torch.nn.functional as F

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
    img = PIL.Image.open(image_path)  
    transform = transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)  


image_path = 'img/coyote.jpg'
image_path = 'img/human.jpg'
image_path = 'img/bldeagle_test.jpg'
# image_path = 'img/bobcat.jpg'

image = preprocess_image(image_path)  

with torch.no_grad():  
    prediction = model(image)

_, predicted_class_index = torch.max(prediction.data, 1)


class_labels = os.listdir('oregon_wildlife')
# predicted_class = class_labels[predicted_class_index.item()]
# print(f'Predicted class: {predicted_class}')

predicted_class = class_labels[predicted_class_index.item()]
if torch.max(prediction.data) > 7.0000:

    print(class_labels)
    print(prediction)
    print(torch.max(prediction.data))
    print(f'Predicted class: {predicted_class}: CORRECT!!!!')

else:
    print(class_labels)
    print(prediction)
    print(torch.max(prediction.data))
    print(f'Predicted class: {predicted_class}: WRONG!!!!!')
    print("ACTUAL CLASSIFICASTION: Null")