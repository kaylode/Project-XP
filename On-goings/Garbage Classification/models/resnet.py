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
        self.model_name = "ResNet34"
        self.optimizer = self.optimizer(self.parameters(), lr= self.lr)

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

    def training_step(self, batch):
        inputs = batch["img"]
        targets = batch["label"]
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        return loss

    def evaluate_step(self, batch):
        inputs = batch["img"]
        targets = batch["label"]
        if self.device:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        outputs = self(inputs) #batchsize, label_dim
        loss = self.criterion(outputs, targets)

        metric_dict = self.update_metrics(outputs, targets)
        
        return loss , metric_dict

    def forward_test(self):
        inputs = torch.rand(1,3,224,224)
        if self.device:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self(inputs)
        return outputs

    def update_metrics(self, outputs, targets):
        metric_dict = {}
        for metric in self.metrics:
            metric.update(outputs, targets)
            metric_dict.update(metric.value())
        return metric_dict

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()