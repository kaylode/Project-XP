import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, padding= 1, stride= 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding= 1, stride= 1)
        self.max_pool2d = nn.MaxPool2d(kernel_size = 2,stride = 2)
        self.fc1 = nn.Linear(3136,256)
        self.fc2 = nn.Linear(256,10)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = x.reshape(x.size(0), -1)    #Flatten image
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

            