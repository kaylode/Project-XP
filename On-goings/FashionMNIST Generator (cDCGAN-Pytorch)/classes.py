import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np 
import matplotlib.pyplot as plt
import functions as f

LATENT_SIZE = 100
ngf = 64
ndf = 64
nc = 1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100,128*8,4,1,0),
            nn.BatchNorm2d(128*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(128*8,128*4,4,2,1),
            nn.BatchNorm2d(128*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(128*4,128*2,4,2,1),
            nn.BatchNorm2d(128*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(128*2,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128,1,4,2,1),
            nn.Tanh()

        )
    def forward(self, x):
        x = self.main(x)
        return x
        

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1,128,4,2,1, bias=False),
            nn.LeakyReLU(0.2, inplace= True),

            nn.Conv2d(128,128*2,4,2,1, bias=False),
            nn.BatchNorm2d(128*2),
            nn.LeakyReLU(0.2, inplace= True),

            nn.Conv2d(128*2,128*4,4,2,1, bias=False),
            nn.BatchNorm2d(128*4),
            nn.LeakyReLU(0.2, inplace= True),

            nn.Conv2d(128*4,128*8,4,2,1, bias=False),
            nn.BatchNorm2d(128*8),
            nn.LeakyReLU(0.2, inplace= True),

            nn.Conv2d(128*8,1,4,1,0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)