import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import pandas as pd 
import matplotlib.pyplot as plt
import cv2 
import os
from tqdm import tqdm
from classes import CNN
import functions as f
from sklearn.model_selection import KFold
import csv
path = os.path.dirname(__file__)


PROCESS_DATA = 0  #Turn off
TRAINING_DATA = 0

#Inititate constants
BATCH_SIZE = 1024
IMG_SIZE = 28
TRAINING_SAMPLES = 48000
VAL_SAMPLES = 12000

device =  torch.device("cuda: 0") if torch.cuda.is_available() else torch.device("cpu")
print(device)

#Data Augmentation for normalizing data
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])

if PROCESS_DATA:
    dataset = datasets.MNIST("data",train=True, transform = data_transforms, download = True)
    trainset, valset = data.random_split(dataset, [TRAINING_SAMPLES, VAL_SAMPLES])
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, num_workers=4)
    


#Training Inititate
model = CNN()
model = model.to(device)
if TRAINING_DATA:
    error = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    model = train(model,error,optimizer,trainloader,valloader)
else:
    model.load_state_dict(torch.load("model/model1.pth"))

f.visualize_model(model)

