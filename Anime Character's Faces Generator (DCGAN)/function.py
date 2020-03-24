import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from PIL import Image
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

def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)

#Make random latent noise vector
def random_noise(size):
    noise = torch.randn(size, LATENT_SIZE, 1 , 1)
    return Variable(noise).to(device)

#Generate truth or false label
def generate_label(size,label):
    if label:
        data = torch.ones(size,1) - 0.1 #Smooth label one-side
    else:
        data = torch.zeros(size,1)
    return Variable(data).view(-1).to(device)


#Train the discriminator
def train_discriminator(D, D_optimizer, error, real_data, fake_data):
    batch_size = real_data.size(0)
    D_optimizer.zero_grad()
    
    #Training on all real data
    real_data = real_data.to(device)
    predict_real = D(real_data).view(-1)
    real_label = generate_label(batch_size, 1)
    loss_real = error(predict_real,real_label)
    loss_real.backward()
    
    
    #Training on all fake data
    fake_data = fake_data.to(device)
    predict_fake = D(fake_data).view(-1)
    fake_label = generate_label(batch_size, 0)
    loss_fake = error(predict_fake, fake_label)
    loss_fake.backward()
    
    D_x = predict_real.mean().item()
    D_g_z1 = predict_fake.mean().item()
    
    D_optimizer.step()
    
    return loss_real.data+loss_fake.data, D_x, D_g_z1

    #Train the generator
def train_generator(D, G_optimizer, error, fake_data):
    batch_size = fake_data.size(0)
    G_optimizer.zero_grad()
    fake_data = fake_data.to(device)
    predict = D(fake_data).view(-1)
    real_label = generate_label(batch_size, 1)
    loss = error(predict, real_label)
    loss.backward()
    G_optimizer.step()
    D_g_z2 = predict.mean().item()
    return loss.data, D_g_z2


#Generate some samples
def generate_samples(G, noise):
    G.eval()
    with torch.no_grad():
        fig = plt.figure(figsize=(15,10))
        generated = G(noise).view(-1,3,IMG_SIZE,IMG_SIZE).data.cpu()
        img = np.transpose(torchvision.utils.make_grid(generated[:],padding =1, normalize=True),(1,2,0))
        plt.imshow(img)
        plt.axis("off")
        plt.title("Generated Images")
        plt.show()
    G.train()


#Start training process
def train(D, G, D_optimizer, G_optimizer, error, dataloader):
    EPOCHS = 50
    D_loss_list = []
    G_loss_list = []
    #Generate fixed test noise
    test_noise = random_noise(16)
    print("Start Training.....")
    for epoch in range(EPOCHS):
        for i, (real_batch,_) in enumerate(dataloader):
            #First train the Discriminator
            batch_size = real_batch.size(0)
            real_data = Variable(real_batch.view(-1,3,IMG_SIZE,IMG_SIZE))
            fake_data = G(random_noise(batch_size)).detach()
            D_loss, D_x, D_g_z1 = train_discriminator(D, D_optimizer, error, real_data, fake_data)
            
            #Then train the Generator
            data = G(random_noise(batch_size))
            G_loss, D_g_z2 = train_generator(D, G_optimizer, error, data)
            
            D_loss_list.append(D_loss)
            G_loss_list.append(G_loss)
            
            if i%50==0:
                print("Epoch: ({}/{}), Batch: ({}/{}), D_Loss: {:.4f}, G_Loss: {:.4f}, D(x): {:.4f}, G(D(z)): {:.4f} / {:.4f}".format(epoch,EPOCHS,i,len(dataloader),D_loss,G_loss,D_x, D_g_z1,D_g_z2))
        torch.save(G.state_dict(), "model/generator.pth")
        torch.save(D.state_dict(), "model/discriminator.pth")
        if epoch%5==0:
            generate_samples(G, test_noise)
    print("Training Completed!")
    return G, D_loss_list, G_loss_list


#Plot learning curves
def plot(D_loss_list, G_loss_list, EPOCHS=None):
    if EPOCHS is None:
        EPOCHS = len(D_loss_list)
    fig = plt.figure(figsize=(35,5))
    fig.add_subplot(1,2,1)
    plt.plot(range(EPOCHS), D_loss_list,label = "Discriminator Loss")
    plt.plot(range(EPOCHS),G_loss_list, color ="orange",label = "Gennerator Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

#Generate final samples
def visualize_final(NUM_PIC=16):
    test_noise = random_noise(NUM_PIC)
    generate_samples(G, test_noise)