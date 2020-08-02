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
from trainer.trainer import Trainer

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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam
    model = ResNet34(NUM_CLASSES, lr = 1e-3, criterion= criterion, optimizer= optimizer, device = device)
    
    trainer = Trainer(model, trainloader, valloader)
    print(trainer)
    
    #trainer.fit(print_per_iter=100)
    loss, metrics = trainer.evaluate_epoch()
    print(loss)
    print(metrics)
