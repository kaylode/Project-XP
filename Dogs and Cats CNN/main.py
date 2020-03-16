import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms, datasets, models
import os
import cv2
from tqdm import tqdm
from sklearn.model_selection import KFold
import function as f
from PIL import Image

path = os.path.dirname(__file__)
device = torch.device("cuda: 0")

TRAINING_DATA = 0

BATCH_SIZE = 64
IMG_SIZE = 224
TRAINING_SAMPLES = 20000
VAL_SAMPLES = 4935

LABEL = {"cats":0, "dogs": 1}

DATA_DIR = 'data/trainingSet' 
TEST_DIR = 'data/testSet'

data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

if __name__ == '__main__':
        if TRAINING_DATA:
                dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
                
                train_datasets ,val_datasets = torch.utils.data.random_split(dataset, [TRAINING_SAMPLES, VAL_SAMPLES]) 
                
                trainloader = data.DataLoader(train_datasets, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
                valloader = data.DataLoader(val_datasets, batch_size=BATCH_SIZE, num_workers=4)

                train_loss_list = []
                val_loss_list = []
                val_acc_list = []

                model = models.resnet18(pretrained=True)
                model = model.to(device)
                for param in model.parameters():
                        param.requires_grad = False
                model.fc = nn.Linear(in_features=512, out_features=2).to(device)
                error = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
                model = f.train(model,error,optimizer,trainloader,valloader)
        else:
                model = models.resnet18(pretrained=True)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 2)
                model = model.to(device)
                model.load_state_dict(torch.load("model/model5.pth"))

        f.visualize_model(model,TEST_DIR)
      

        
