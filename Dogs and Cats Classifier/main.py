import numpy as np 
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
from tqdm import tqdm

path = os.path.dirname(__file__)
PREPROCESS_DATA = 0

if PREPROCESS_DATA:
    DATA = "data/trainingSet"
    LABEL = {"cats":0, "dogs": 1}
    training_data = []
    count = 0
    for name,label in LABEL.items():
        path1 = os.path.join(path,DATA,name)
        try:
            for f in tqdm(os.listdir(path1)):
                path2 = os.path.join(path1,f)
                img = cv2.imread(path2,0)
                training_data.append([np.array(img),label])
        except Exception as e:
            pass
    
    np.save('data_saves/training_data1.npy',training_data)
else:
    training_data = np.load("data_saves/training_data1.npy", allow_pickle=1)



