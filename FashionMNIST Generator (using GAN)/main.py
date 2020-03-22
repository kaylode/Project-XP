import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np 
import matplotlib.pyplot as plt
import functions as f
import cv2
from classes import Discriminator, Generator, weights_init
from PIL import Image

LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
device = torch.device("cuda: 0")
BATCH_SIZE = 64
EPOCHS = 5
IMG_SIZE = 64

data_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

TRAINING = 0






if __name__ =="__main__":
    D = Discriminator().to(device)
    G = Generator().to(device) 
    D.apply(weights_init)
    G.apply(weights_init)

    if TRAINING:
        dataset = datasets.FashionMNIST("data",train = True,transform=data_transform,download = True)
        trainloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers = 4, shuffle=True)
        
        D_optimizer = torch.optim.Adam(D.parameters(), lr= 0.0002, betas=(0.5, 0.999))
        G_optimizer = torch.optim.Adam(G.parameters(), lr = 0.0002, betas=(0.5, 0.999))
    
        error = nn.BCELoss()

        G = f.train(EPOCHS,D,G,D_optimizer,G_optimizer,error,trainloader)
        
    else:
        G.load_state_dict(torch.load("model/generator-dcgan.pth"))
        test_noise = f.random_noise(16)
        f.generate_img(G,test_noise,1)



  
    


"""

    test_noise = f.random_noise(10)
    test_img = G(test_noise).view(-1,1,28,28).data
    test_img = test_img.cpu()
    plt.subplot(2,5,1)
    for i,img in enumerate(test_img):
        plt.subplot(2,5,i+1)
        f.imshow(img)
    plt.show()

"""
   