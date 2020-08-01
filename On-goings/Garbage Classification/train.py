from datasets.image_classification import ImageClassificationDataset
import numpy as np
import random
import torch.utils.data as data
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from models.resnet import ResNet34
from losses.smoothceloss import smoothCELoss
from metrics.classification.accuracy import AccuracyMetric

transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def calc_accuracy(preds, y):
    correct = (preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

    

def train_epoch(model, optimizer,criterion,  trainiter, print_per_iter = 500):
    model.train()
    epoch_loss = 0
    iter_loss = 0
   
    for i, batch in enumerate(tqdm(trainiter)):
        optimizer.zero_grad()

        inputs = batch["img"]
        targets = batch["label"]
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs) #batchsize, label_dim
        loss = criterion(outputs, targets)

        loss.backward()
        
        optimizer.step()
        

        epoch_loss += loss.item()
        iter_loss += loss.item()
     

        if (i % print_per_iter == 0 or i == len(trainiter) - 1) and i != 0:
            print("\tIterations: [{}|{}] | Train loss: {:10.4f}".format(i+1, len(trainiter), iter_loss/ print_per_iter))
            iter_loss = 0
    return epoch_loss / len(trainiter)

def evaluate_epoch(model, criterion, valiter):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    acc = AccuracyMetric()
    with torch.no_grad():
          for batch in tqdm(valiter):

                inputs = batch["img"]
                targets = batch["label"]
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs) #batchsize, label_dim
                loss = criterion(outputs, targets)

                acc.update(outputs, targets)
             
                epoch_loss += loss
    epoch_acc = acc.value()

    return epoch_loss / len(valiter), epoch_acc / len(valiter)


if __name__ == "__main__":
    trainset = ImageClassificationDataset("datasets/garbage_train", transforms= transforms)
    valset = ImageClassificationDataset("datasets/garbage_val", transforms= transforms)
    print(trainset)
    print(valset)
    
    NUM_CLASSES = len(trainset.classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Dataloader
    BATCH_SIZE = 2
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    valloader = data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True)



    """  
    #  Create model
    model = models.resnet34(pretrained = True)
    for params in model.parameters():
        params.requires_grad = False

    model.fc = nn.Linear(512, NUM_CLASSES)
    model = model.to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
    #print(model)

    EPOCHS = 10
    print("Start training...")

    best_val_acc = 0.0
    best_val_loss = 100

    for epoch in range(0,EPOCHS):
        train_loss = train_epoch(model,optimizer, criterion, trainloader, print_per_iter=100)
        val_loss, val_acc = evaluate_epoch(model, criterion, valloader)
        
        print("Epoch: [{}/{}] |  Train loss: {:10.4f} | Val Loss: {:10.4f} | Val Acc: {:10.4f}".format(epoch+1, EPOCHS, train_loss, val_loss, val_acc))
        #torch.save(model.state_dict(), "../drive/My Drive/model BiGRU/BiGRU-{}-{:10.4f}.pth".format(epoch+1, val_acc))

    print("Training Completed!")"""
    
    EPOCHS = 10
    model = ResNet34(n_classes = NUM_CLASSES, device = device)
    criterion = smoothCELoss(device= device)
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
    for epoch in range(0,EPOCHS):
        train_loss = train_epoch(model,optimizer, criterion, trainloader, print_per_iter=100)
        val_loss, val_acc = evaluate_epoch(model, criterion, valloader)
        
    
