# Jake Eckfeldt
# 11688261 CPTS 528

# models.py
# File for model logic


import torch
import torch.nn as nn
import torch.nn.functional as F

# convolutional neural network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        
        # pool reduces image size
        self.pool = nn.MaxPool2d(2, 2)

        # dropout shut off nodes randomly to avoid memorizing
        self.dropout = nn.Dropout(0.25)

        # final mapping
        self.fc1 = nn.Linear(128 * 4 * 4, 10)

    # function to step through network
    def forward(self, x):
        # first layer
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # second layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # third layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # flatten to 1 dimension
        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)

        return x
