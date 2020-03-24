import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import math
import torchvision
from torchvision import datasets, transforms

BATCH_SIZE = 1024
IMG_SIZE = 28
TRAINING_SAMPLES = 48000
VAL_SAMPLES = 12000

#Data Augmentation for normalizing data
data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ])

device = torch.device("cuda: 0")
#Forward Propagation for both training and validation
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

    #Plot loss curve
def plot(train_loss_list, val_loss_list, val_acc_list, EPOCH):
    fig = plt.figure(figsize=(15,5))
    fig.add_subplot(1,2,1)
    plt.plot(range(EPOCH),train_loss_list)
    plt.plot(range(EPOCH),val_loss_list)
    plt.title("Loss per epoch")
    fig.add_subplot(1,2,2)
    plt.plot(range(EPOCH),val_acc_list)
    plt.title("Validation Accuracy")
    plt.show()

#Training process
def train(model, error, optimizer, trainloader, valloader):
    EPOCHS = 10
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    print("Start training process....")
    for epoch in range(EPOCHS):
        train_loss, val_acc, val_loss = forwardprop(model,error,optimizer,trainloader,valloader,True)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print("Epoch: ({}/{}) | Training Loss: {:.2f} | Validation Loss: {:.2f} | Validation Accuracy: {:.2f}".format(epoch+1,EPOCHS,train_loss,val_loss,val_acc))
    #Save model
    torch.save(model.state_dict(), "model/model2.pth")
    print("Training Completed! Model has been saved")
    #Plot loss
    print("Ploting loss...")
    plot(train_loss_list,val_loss_list,val_acc_list,EPOCHS)
    return model



#Predict input images
def predict(model, images):
    model.eval()
    with torch.no_grad():
        test = Variable(images).view(-1,1,IMG_SIZE,IMG_SIZE)
        test = test.to(device)
        outputs = model(test)
        prediction = outputs.data
        prediction = torch.max(outputs.data, 1)[1]
    return prediction.cpu().numpy()[0]

def preprocess_image(img):
    #Apply threshold
    thresh = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    #Find countor, bounding box
    cnts, tmp = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 
    if len(cnts) == 0:
        print("Sorry No contour Found.")
    else:
        x,y,w,h = cv2.boundingRect(cnts[0])
        img_crop = img[y:(y+h), x:(x+w)]
        
        DIGITS_SIZE = 20
        #Resize image, keep ratio
        a,b=h,w
        d = int(max(h,w)/DIGITS_SIZE)
        if h > DIGITS_SIZE:
            a = int(h/d)
        if w > DIGITS_SIZE:
            b = int(w/d)
        a = 5 if a <= 0 else a
        b = 5 if b <= 0 else b
        img_crop = cv2.resize(img_crop,(b,a))
    
        horizontal = 28 - b
        vertical = 28 - a
        top = bot = int(vertical/2)
        left = right = int(horizontal/2)
        #Padding the image
        img_padding = cv2.copyMakeBorder(img_crop,top,bot,left,right,cv2.BORDER_CONSTANT)
        #Dilate
        kernel = np.ones((2,2), np.uint8) 
        img_dilation = cv2.dilate(img_padding,kernel=kernel,iterations = 1)
        
        img = cv2.resize(img_dilation,(28,28))
    return img

#Predict and visualize some test images
def visualize_model(model,NUM_PIC=16):
    testset = datasets.MNIST("data",train=False, transform = data_transforms, download = False)
    a = np.random.randint(100)
    fig = plt.figure(figsize=(10,3))
    for i in range(a,a+NUM_PIC):
        test_img,_ = testset[i]
        result = predict(model,test_img)
        fig.add_subplot(2,NUM_PIC/2,i-a+1)
        img = test_img.numpy().transpose((1, 2, 0))
        img = img.squeeze()
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(result)
    plt.show()