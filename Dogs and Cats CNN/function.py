import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torchvision import transforms, datasets
import os
from PIL import Image

path = os.path.dirname(__file__)
device = torch.device("cuda: 0")

BATCH_SIZE = 64
IMG_SIZE = 224
TRAINING_SAMPLES = 20000
VAL_SAMPLES = 4946


def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    
def forwardprop(model, error, optimizer, trainloader, valloader=None, training = False):
    correct = 0
    total = 0
    total_loss = 0
    if training:
        model.train()
    else:
        model.eval()
    for (train, labels) in trainloader:
        train = train.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad()
        outputs = model(train)
        predicted = torch.max(outputs.data, 1)[1]
        loss = error(outputs, labels)
        if training:
            loss.backward()
            optimizer.step()
        else:
            correct += (predicted==labels).sum()
        total += len(labels)
        total_loss += loss.data
    if training:
        with torch.no_grad():
            val_acc, val_loss = forwardprop(model = model, error = error, optimizer = optimizer, trainloader = valloader)
    else:
        accuracy = correct*1.0/total

    if training:    
        return total_loss, val_acc, val_loss
    else:
        return accuracy, total_loss

def train(model, error, optimizer, trainloader, valloader):
    EPOCHS = 10
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []

    for epoch in range(EPOCHS):
        train_loss, val_acc, val_loss = forwardprop(model,error,optimizer,trainloader,valloader,True)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print("Epoch: {} | Training Loss: {:.2f} | Validation Loss: {:.2f} | Validation Accuracy: {:.2f}".format(epoch+1,train_loss,val_loss,val_acc))
    torch.save(model.state_dict(), "model/model5.pth")
    plot(train_loss_list,val_loss_list,val_acc_list,EPOCHS)
    return model

def predict(model, images):
    model.eval()
    with torch.no_grad():
        test = Variable(images).view(-1,3,IMG_SIZE,IMG_SIZE)
        test = test.to(device)
        outputs = model(test)
        prediction = outputs.data
        #prediction = F.softmax(prediction, dim=1)
        prediction = torch.max(outputs.data, 1)[1]
    return prediction.cpu().numpy()[0]

def plot(train_loss_list, val_loss_list, val_acc_list, EPOCH):
    plt.subplot(2,1,1)
    plt.plot(range(EPOCH),train_loss_list)
    plt.plot(range(EPOCH),val_loss_list)
    plt.title("Loss per epoch")
    plt.subplot(212)
    plt.plot(range(EPOCH),val_acc_list)
    plt.title("Validation Accuracy")
    plt.show()

def visualize_model(model,TEST_DIR, NUM_PIC = 10):

    test_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    test_path = os.listdir(os.path.join(path,TEST_DIR+"/test"))
    a = np.random.randint(2000)
    plt.subplot(2,NUM_PIC/2,1)
    for id, i in enumerate(test_path[a:a+10]):
        img = Image.open(os.path.join(path,TEST_DIR+"/test/"+i))
        img2 = test_transforms(img)
        result = predict(model,img2)
        plt.subplot(2,NUM_PIC/2,id+1)
        plt.imshow(img)
        plt.title("CAT" if result==0 else "DOG")
    plt.show()