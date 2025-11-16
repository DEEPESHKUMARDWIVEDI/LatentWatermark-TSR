# models/tsr_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TSRNet(nn.Module):
    def __init__(self, num_classes=43):
        super(TSRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop2d = nn.Dropout2d(0.25)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop2d(x)

        x = self.gap(x).view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# class TSRNet(nn.Module):
#     """Lightweight CNN for Traffic Sign Recognition (32x32 inputs)"""
#     def __init__(self, num_classes=43):
#         super(TSRNet, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # output: 32x32x32
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # output: 64x32x32
#         self.pool = nn.MaxPool2d(2, 2)               # halves to 64x16x16
#         self.conv3 = nn.Conv2d(64, 64, 3, padding=1) # output: 64x16x16
#         self.pool2 = nn.MaxPool2d(2, 2)              # halves to 64x8x8
        
#         # Fully connected layers
#         self.fc1 = nn.Linear(64 * 16 * 16, 256)
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool2(x)
#         x = x.view(x.size(0), -1)  # flatten
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
