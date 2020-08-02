import os
import torch.nn as nn
import torch
from tqdm import tqdm

class Trainer(nn.Module):
    def __init__(self, 
                model, 
                trainloader, 
                valloader,
                checkpoint_path = 'checkpoint'):

        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = model.optimizer
        self.criterion = model.criterion
        self.trainloader = trainloader
        self.valloader = valloader
        self.metrics = model.metrics #list of metrics
        self.checkpoint_path = checkpoint_path

    def fit(self, num_epochs = 10 ,print_per_iter = None):
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)
        self.num_epochs = num_epochs
        if print_per_iter is not None:
            self.print_per_iter = print_per_iter
        else:
            self.print_per_iter = int(len(self.trainloader)/10)
        

        print("Start training for {} epochs...".format(num_epochs))
        for epoch in range(self.num_epochs):
            train_loss = self.training_epoch()
            val_loss, val_metrics = self.evaluate_epoch()
            print("Epoch: [{}/{}] |  Train loss: {:10.5f} | Val Loss: {:10.5f} | Val Acc: {:10.5f}".format(epoch+1, num_epochs, train_loss, val_loss, val_metrics["acc"]))
            
            self.save_checkpoint()
        print("Training Completed!")


    def save_checkpoint(self):
        torch.save(self.model.state_dict(), "{}/checkpoint-{}-{:10.5f}.pth".format(self.checkpoint_path,epoch+1, val_loss))


    def training_epoch(self):
        self.model.train()
        epoch_loss = 0
        running_loss = 0
    
        for i, batch in enumerate(tqdm(self.trainloader)):
            self.optimizer.zero_grad()
            loss = self.model.training_step(batch)
            loss.backward() 
            self.optimizer.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
        
            if (i % self.print_per_iter == 0 or i == len(self.trainloader) - 1) and i != 0:
                print("\tIterations: [{}|{}] | Training loss: {:10.4f}".format(i+1, len(self.trainloader), running_loss/ self.print_per_iter))
                running_loss = 0
        return epoch_loss / len(self.trainloader)


    def evaluate_epoch(self):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        metric_dict = {}
        with torch.no_grad():
            for batch in tqdm(self.valloader):
                loss, metrics = self.model.evaluate_step(batch)
                epoch_loss += loss
                metric_dict.update(metrics)
        self.model.reset_metrics()

        return epoch_loss / len(self.valloader), metric_dict

    def forward_test(self):
        self.model.eval()
        outputs = self.model.forward_test()
        print("Feed forward success, outputs's shape: ", outputs.shape)

    def __str__(self):
        s1 = "Model name: " + self.model.model_name
        s2 = f"Number of trainable parameters:  {self.model.trainable_parameters():,}"
       
        s3 = "Loss function: " + str(self.criterion)[:-2]
        s4 = "Optimizer: " + str(self.optimizer)
        s5 = "Training iterations per epoch: " + str(len(self.trainloader))
        s6 = "Validating iterations per epoch: " + str(len(self.valloader))
        return "\n".join([s1,s2,s3,s4,s5,s6])