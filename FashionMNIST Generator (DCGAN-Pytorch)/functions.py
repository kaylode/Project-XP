import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np 
import matplotlib.pyplot as plt
import cv2
from PIL import Image
LABELS = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
device = torch.device("cuda: 0")

BATCH_SIZE = 62
IMG_SIZE = 64
LATENT_SIZE = 100



def visualize(dataloader):
    training_img = next(iter(dataloader))
    plt.figure(figsize=(4,4))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(torchvision.utils.make_grid(training_img[0][:16],padding=2, normalize=True),(1,2,0)))
    plt.show()

def imshow(img, label=None):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.5])
    std = np.array([0.5])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    img = np.reshape(img,(IMG_SIZE,IMG_SIZE))
    plt.imshow(img, cmap="gray")
    if label:
        plt.title(LABELS[label])
    
def random_noise(size):
    noise = torch.randn(size,LATENT_SIZE,1,1)
    noise = Variable(noise).to(device)
    return noise

def generate_labels(size, label):
    if label:
        data = torch.ones(size,1)-0.1
    else:
        data = torch.zeros(size,1)
    data = Variable(data).view(-1).to(device)
    return data


def train_discriminator(D, optimizer, error, real_data, fake_data):
    batch_size = real_data.size(0)
    optimizer.zero_grad()

    real_data = real_data.to(device)
    fake_data = fake_data.to(device)

    prediction_real = D(real_data).view(-1)
    real_label = generate_labels(batch_size,1)
    loss_real = error(prediction_real,real_label)
    loss_real.backward()

    prediction_fake = D(fake_data).view(-1)
    fake_label = generate_labels(batch_size,0)
    loss_fake = error(prediction_fake, fake_label)
    loss_fake.backward()

    D_x = prediction_real.mean().item()
    D_g_z1 = prediction_fake.mean().item()
    optimizer.step()

    return loss_real.data + loss_fake.data, D_x, D_g_z1

def train_generator(D, optimizer, error, data):
    batch_size = data.size(0)
    optimizer.zero_grad()
    data = data.to(device)
    prediction = D(data).view(-1)
    label = generate_labels(batch_size, 1)
    loss = error(prediction, label)
    loss.backward()
    D_g_z2 = prediction.mean().item()
    optimizer.step()
    return loss.data, D_g_z2

def plot(D_loss_list, G_loss_list, EPOCHS):
    plt.subplot(2,1,1)
    plt.plot(range(EPOCHS), D_loss_list)
    plt.title("Discriminator Loss")
    plt.subplot(212)
    plt.plot(range(EPOCHS),G_loss_list)
    plt.title("Gennerator Loss")
    plt.show()

def imsave(img,label):
    img = np.array(img)
    plt.imsave("generated/"+str(label)+".jpg",img)
    
def generate_img(G,noise,epoch):
    G.eval()
    
    with torch.no_grad():
        test_img = G(noise).view(-1,1,IMG_SIZE,IMG_SIZE).data.cpu()
        img = np.transpose(torchvision.utils.make_grid(test_img[:],padding=2, normalize=True),(1,2,0))
        img = np.array(img)
        #imsave(img,epoch)
        plt.imshow(img)
        plt.show()
    G.train()

    

def train(EPOCHS, D, G, D_optimizer, G_optimizer, error, trainloader):
    D_loss_list = []
    G_loss_list = []
    
    print("Start Training.....")
    test_noise = random_noise(16)
    for epoch in range(EPOCHS):
        for i, (real_batch,_) in enumerate(trainloader):  
            batch_size = real_batch.size(0)
            fake_data = G(random_noise(batch_size)).detach()
            real_data = Variable(real_batch.view(-1,1,IMG_SIZE,IMG_SIZE))
            D_loss,D_x, D_g_z1 = train_discriminator(D,D_optimizer,error,real_data,fake_data)
            data = G(random_noise(batch_size))
            G_loss, D_g_z2 = train_generator(D,G_optimizer,error,data)
            if i%50 == 0 :
                print("Epoch: ({}/{}), Batch: ({}/{}), D_Loss: {:.4f}, G_Loss: {:.4f}, D(x): {:.4f}, G(D(z)): {:.4f} / {:.4f}".format(epoch,EPOCHS,i,len(trainloader),D_loss,G_loss,D_x, D_g_z1,D_g_z2))
            D_loss_list.append(D_loss)
            G_loss_list.append(G_loss)   
        torch.save(G.state_dict(), "model/generator-dcgan.pth")
        torch.save(D.state_dict(), "model/discriminator-dcgan.pth")
        generate_img(G,test_noise,epoch)
        
    
    plot(D_loss_list,G_loss_list,len(D_loss_list))
    return G
    

