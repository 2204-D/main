import cv2
import serial
import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from torchvision import transforms
import time
from PIL import Image
import random

print("Imports successful")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
transform = transforms.ToTensor()
arduinoData = serial.Serial('COM6', 9600)

# important variables #
resize_thing = (512, 384)

all_classes = ['glass', 'metal', 'paper', 'plastic']
degrees = {'glass': '80 150-1', 'metal': '30 150-1', 'paper': '100 45-1', 'plastic': '40 30-1'}
            # kablo sol yakın | # kablo sağ uzak | # kablo sağ yakın | # kablo sol uzaka
# kablolara önden bakarken

model_location = "trashnet_filtered_epoch_3.pt"
num_classes = 4


# important variables #

class TrashNetWithResNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(TrashNetWithResNet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet_layers = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.resnet_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def predict_clas(foto):
    foto_tensor = transform(foto)
    output = model(foto_tensor.unsqueeze(0).cpu())
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    predicted_classes = torch.argsort(probabilities, descending=True)
    return predicted_classes


def spin(degree):
    arduinoData.write(degree.encode())


model = TrashNetWithResNet()
model.load_state_dict(torch.load(model_location, map_location=torch.device('cpu')), strict=False)

while True:
    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, resize_thing)

    cv2.imshow('Webcam', resized_frame)

    foto = Image.fromarray(resized_frame)

    predicted_classes = predict_clas(foto)

    print(f"Predicted Class: {all_classes[predicted_classes[0]]}")

    #spin(degrees[all_classes[predicted_classes[0]]])
    a = random.choice(all_classes)
    print(a)
    spin(degrees[a])

    if cv2.waitKey(1) & 0xFF == ord('q'): break

    time.sleep(1)  # temp sleep half second to give servo time

cap.release()
cv2.destroyAllWindows()

# random is better for testing to spin the servo (for now)
# check if last time more than half second

# last = time.time()
#
# if time.time() - last > 0.5:
#     last = time.time()
#     spin(degrees[random.choice(all_classes)])
