#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
from torchvision import transforms, datasets, models
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

# In[35]:


def data_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


# In[36]:


def imshow(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_show = img.cpu().numpy().squeeze().transpose((1,2,0))
    img_show = (img_show * std+mean)
    img_show = np.clip(img_show,0,1)
    return img_show


# In[37]:


class SVHNDataset(data.Dataset):
    def __init__(self, root, annotation, transforms=None, device = device):
        self.root = root
        self.transforms = transforms
        self.annotation = annotation
        self.transforms = transforms
        self.ids = os.listdir(root)
        self.classes = [i for i in range(0,11)]
        self.device = device
        with open(self.annotation, 'r') as anno:
            json_file = json.load(anno)
        self.json_file = json_file
    def __getitem__(self, index):
        obj_name = self.json_file[index]['filename']
        img_id = Image.open(os.path.join(self.root,obj_name))
            
        obj_boxes = self.json_file[index]['boxes']
        
        num_obj = len(obj_boxes)
        boxes = []
        labels = []
        for i in range(num_obj):
            xmin =  obj_boxes[i]['left']
            ymin =  obj_boxes[i]['top']
            xmax =  xmin + obj_boxes[i]['width']
            ymax =  ymin + obj_boxes[i]['height']
            label = obj_boxes[i]['label']
            boxes.append([xmin,ymin,xmax,ymax])
            labels.append(label)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.LongTensor(labels)
        index = torch.Tensor([index])
        iscrowd = torch.zeros((num_obj,), dtype=torch.int64)
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = index
        my_annotation["iscrowd"] = iscrowd
        my_annotation["area"] = area
        
        if self.transforms is not None:
            img = self.transforms()(img_id)
        
        return img, my_annotation
    
    def __len__(self):
        return len(self.ids)
    


# In[38]:


def collate_fn(batch):
    return tuple(zip(*batch))


# In[39]:


#Visualize 10 training samples
def visualize_samples():
    fig = plt.figure(figsize=(15,10))
    plt.title("Training Samples")
    for id,(img,anno) in enumerate(dataloader): 
        if id==9:
            break
        ax = fig.add_subplot(3,3,id+1)
        for label, box in zip(anno[0]["labels"], anno[0]["boxes"]):
            rect = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,ec="red",lw=3)
            plt.text(box[0],box[1]-3,int(label.numpy()),color="red",fontsize=15, fontweight='bold')
            ax.add_patch(rect)
        img = imshow(img[0])
        plt.imshow(img)
    plt.show()
#visualize_samples()


# In[40]:


def instance_segmentation_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

#print(model)


# In[41]:



def train(model,dataloader):
    EPOCHS = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_list = []
    print("Start training...")
    for epoch in range(EPOCHS):
        model.train()
        for ids, (imgs,anno) in enumerate(dataloader):
        
            imgs = list(image.to(device) for image in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in anno]
            
            
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            loss_list.append(losses)
            
            print(f'Epoch: [{ids}/{len(dataloader)}], Loss: {losses}')
            if ids %3000 == 0:
                torch.save(model.state_dict(), "model/model.pth")
    torch.save(model.state_dict(), "model/model.pth")
    print("Training Completed!")
    
    return model, loss_list


# In[42]:


if __name__ == "__main__":
    
    ANNOTATION_DIR = {
    "train": "annotation/train.json",
    "val": "annotation/val.json"
    }

    DATA_DIR = {
        "train": "data/train",
        "val": "data/val"
    }
  
    TRAIN_DATA = 1
    dataset = SVHNDataset(DATA_DIR["train"],ANNOTATION_DIR["train"],transforms=data_transform, device=device)
    dataloader = data.DataLoader(dataset,batch_size=1,shuffle=True, collate_fn=collate_fn, num_workers = 3)
    CLASSES = dataset.classes
    NUM_CLASSES = len(CLASSES)
    TRAINING_SAMPLES =len(dataset)
    model = instance_segmentation_model(NUM_CLASSES)
    model = model.to(device)
    if TRAIN_DATA:
        model, loss_list = train(model,dataloader)
    
    

