import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import function as f
from classes import Discriminator, Generator, weights_init
import numpy as np 
import matplotlib.pyplot as plt

PROCESS_DATA = 1
TRAINING_DATA =1

DATA_DIR = "data/trainingSet"
IMG_SIZE = 64
BATCH_SIZE = 64
LATENT_SIZE = 100


data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if __name__ == "__main__":
    if PROCESS_DATA:
        dataset = datasets.ImageFolder(DATA_DIR, transform= data_transform)
        dataloader = data.DataLoader(dataset,batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
        
    D = Discriminator().to(device)
    G = Generator().to(device)
    D.apply(weights_init)
    G.apply(weights_init)

    if TRAINING_DATA:
        D_optimizer = torch.optim.Adam(D.parameters(), lr = 0.0002, betas= (0.5, 0.999))
        G_optimizer = torch.optim.Adam(G.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        error = nn.BCELoss()
        G, D_loss_list, G_loss_list = f.train(D, G, D_optimizer, G_optimizer, error, dataloader)
        f.plot(D_loss_list, G_loss_list)
    else:
        G.load_state_dict(torch.load("model/generator.pth"))

        visualize_final()