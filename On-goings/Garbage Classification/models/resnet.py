from .base_model import BaseModel
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms



class ResNet34(BaseModel):
    def __init__(self, n_classes, **kwargs):
        super(ResNet34, self).__init__(**kwargs)
        self.model = models.resnet34(pretrained = True)
        self.n_classes = n_classes

        if self.freeze:
            for params in self.model.parameters():
                params.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, n_classes)

        if self.device:
            self.model.to(self.device)
            
    def forward(self, x):
        return self.model(x)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def unfreeze(self):
        for params in self.model.parameters():
            params.requires_grad = True

    def parameters(self):
        return self.model.parameters()
    
