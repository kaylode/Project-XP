
import torch
import torch.nn as nn
import torch.utils.data as udata
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw 
import cv2
import csv
import os
from classes import CNN
from tqdm import tqdm

def visualize_train_data():
    a = np.random.randint(30000)
    fig = plt.figure(figsize=(20,30))
    for id, (name,label,bbox) in enumerate(annotation[a:a+10]):
        fig.add_subplot(5,2,id+1)

        img = cv2.imread(os.path.join(TRAIN_DIR,name))
        h,w,_ = img.shape

        targetSize = IMG_SIZE

        x_scale = targetSize/w
        y_scale = targetSize/h

        bx,by,bw,bh = bbox
        bx = int(np.round(bx * x_scale))
        by = int(np.round(by * y_scale))
        bw = int(np.round(bw * x_scale))
        bh = int(np.round(bh * y_scale))

        img = cv2.resize(img, (targetSize, targetSize))

        cv2.rectangle(img,(bx,by),(bx+bw,by+bh), color = (255,0,0),thickness = 1)
        plt.imshow(img)

    plt.show() 

class CNN(nn.Module):
    def __init__(self, inp, d, out):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp,d,kernel_size=3,stride = 1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(d,d*2,kernel_size=3,stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(d*2,d*4,kernel_size=3,stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(d*4,d*8,kernel_size=3,stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(d*8,d*8,kernel_size=3,stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2,2)
        
        )
      
        self.fc = nn.Sequential(
            nn.Linear(16384,256),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            
            nn.Linear(256,4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            
            nn.Linear(4096,256),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            
            nn.Linear(256,out)
        )
    
        
    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

def make_data():
    processed_data = []
    for id, (name,label,bbox) in tqdm(enumerate(annotation)):
        img = cv2.imread(os.path.join(TRAIN_DIR,name),0)
        h,w = img.shape
        label = [int(float(i)) for i in label]
        x_scale = IMG_SIZE/w
        y_scale = IMG_SIZE/h

        bx,by,bw,bh = bbox
        bx = int(np.round(bx * x_scale))
        by = int(np.round(by * y_scale))
        bw = int(np.round(bw * x_scale))
        bh = int(np.round(bh * y_scale))
        bb = (bx,by,bw,bh)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        processed_data.append([np.array(img), bb, label])
    return processed_data

def train(model, optimizer, error, trainloader):
    model.train()
    for (img,label) in trainloader:
        img_tensor = Variable(torch.Tensor(img)).view(-1,1,IMG_SIZE,IMG_SIZE).to(device)
        label_tensor = Variable(torch.LongTensor(label)).to(device)
        outputs = model(img_tensor)
        loss = error(outputs, label_tensor)
        loss.backward()
        optimizer.step()
    return loss.data


def start_process(model, optimizer, error, trainloader):
    print("Start Training...")
    loss_list = []
    for epoch in range(100):
        loss = train(model,optimizer,error,trainloader)
        loss_list.append(loss.data)
        print("Epoch: {},  Loss: {}".format(epoch+1,loss.data))
    return loss_list

if __name__ == "__main__":
    device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "data/data.npy"
    IMG_SIZE = 256
    BATCH_SIZE = 32

    np_data = np.load(DATA_DIR, allow_pickle=True)

    traindata = [i[0] for i in np_data]
    trainbbox = [i[1] for i in np_data]

    traindata_tensor = torch.from_numpy(np.array(traindata))
    trainbbox_tensor = torch.from_numpy(np.array(trainbbox))
    trainset = udata.TensorDataset(traindata_tensor,trainbbox_tensor)
    trainloader = udata.DataLoader(trainset, batch_size=BATCH_SIZE)

    model = CNN(1,32,4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.002)
    error = nn.MSELoss()

    start_process(model, optimizer, error, trainloader)



