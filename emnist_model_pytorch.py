import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from torchsummary import summary

# Define the enhanced neural network for EMNIST
class EnhancedNet(nn.Module):
    def __init__(self):
        super(EnhancedNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 47)  # 47 classes for EMNIST balanced
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load EMNIST data with augmentation
def get_emnist_data():
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.EMNIST(root='D:/SVNIT/Semester-5/CISMR/RA_AIR_24', split='balanced', train=True, download=True, transform=transform)
    testset = torchvision.datasets.EMNIST(root='D:/SVNIT/Semester-5/CISMR/RA_AIR_24', split='balanced', train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)
    return trainloader, testloader

# Load class mapping from .txt file
def load_class_mapping(file_path):
    class_mapping = {}
    with open(file_path, 'r') as f:
        for line in f.readlines():
            class_index, char_code = line.strip().split()
            class_index = int(class_index)
            character = chr(int(char_code))
            class_mapping[class_index] = character
    return class_mapping

# Train the enhanced model for EMNIST
def train_model(trainloader, testloader, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Check accuracy on test data
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}, Accuracy: {accuracy}')
        if accuracy > 0.99:
            print("Reached 99% accuracy so cancelling training!")
            break
    
    return model

# Predict digit using image passed in
def predict(model, img, class_mapping):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    output = model(img)
    _, predicted = torch.max(output, 1)
    class_index = predicted.item()
    if class_index in class_mapping:
        return class_mapping[class_index]
    else:
        return str(class_index)  # Fallback to index if mapping not found

# OpenCV part

startInference = False
def ifClicked(event, x, y, flags, params):
    global startInference
    if event == cv2.EVENT_LBUTTONDOWN:
        startInference = not startInference

threshold = 100
def on_threshold(x):
    global threshold
    threshold = x

def start_cv(model, class_mapping):
    global threshold
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('background')
    cv2.setMouseCallback('background', ifClicked)
    cv2.createTrackbar('threshold', 'background', 150, 255, on_threshold)
    background = np.zeros((480, 640), np.uint8)
    frameCount = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    while True:
        ret, frame = cap.read()
        if startInference:
            frameCount += 1
            frame[0:480, 0:80] = 0
            frame[0:480, 560:640] = 0
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thr = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY_INV)
            resizedFrame = thr[240-75:240+75, 320-75:320+75]
            background[240-75:240+75, 320-75:320+75] = resizedFrame
            iconImg = cv2.resize(resizedFrame, (28, 28))
            iconImg = iconImg.astype(np.float32) / 255.0
            res = predict(model, iconImg, class_mapping)
            if frameCount == 5:
                background[0:480, 0:80] = 0
                frameCount = 0
            cv2.putText(background, res, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.rectangle(background, (320-80, 240-80), (320+80, 240+80), (255, 255, 255), thickness=3)
            cv2.imshow('background', background)
        else:
            cv2.imshow('background', frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    model = None
    class_mapping = load_class_mapping('D:/SVNIT/Semester-5/CISMR/RA_AIR_24/EMNIST/raw/emnist-balanced-mapping.txt')
    if os.path.exists('Trial-9/cd-model.pth'):
        model = EnhancedNet()
        model.load_state_dict(torch.load('Trial-9/cd-model.pth'))
        print('Loaded saved model.')
    else:
        trainloader, testloader = get_emnist_data()
        epochs = 100
        model = train_model(trainloader, testloader, epochs)
        torch.save(model.state_dict(), 'Trial-9/cd-model.pth')
        
    start_cv(model, class_mapping)
    summary(model, (1, 28, 28))

if __name__ == '__main__':
    main()
