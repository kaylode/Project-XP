import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import pandas as pd 
import matplotlib.pyplot as plt
import cv2 
import os
from tqdm import tqdm
from classes import CNN
import functions as f
from sklearn.model_selection import KFold

path = os.path.dirname(__file__)

PROCESS_IMAGES = 0
READ_DATA = 0
TRAIN_DATA = 0
if PROCESS_IMAGES:
    img = cv2.imread(path+"/digits.png", 0)
    cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
    id = 0
    for i in range(50):
        for j in range(100):
            cv2.imwrite("data/"+str(id)+".png",cells[i][j])
            id+=1

if READ_DATA:
    DATA = "data/trainingSet"
    training_data = []
    count = 0
    for labels in range(10):
        path1 = os.path.join(path,DATA,str(labels))
        for f in tqdm(os.listdir(path1)):
            path2 = os.path.join(path1,f)
            img = cv2.imread(path2,0)
            training_data.append([np.array(img), labels])
            count+=1
    np.random.shuffle(training_data)
    np.save("training_data2.npy", training_data)
else:
    training_data = np.load("training_data2.npy",allow_pickle=1)


device = torch.device("cuda: 0")


best_acc_list=[]
best_loss_list=[]
fold_val_score =[]
if TRAIN_DATA:
    

    X = torch.Tensor([i[0] for i in training_data]).view(-1,28,28)
    X = X/255.0
    y = torch.Tensor([i[1] for i in training_data]).type(torch.LongTensor)

    """
    val_size = 4200
    X_train = X[:-val_size]
    y_train = y[:-val_size]

    X_val = X[-val_size:]
    y_val = y[-val_size:]
    """
    kf = KFold(n_splits = 10)
    best_val_score = 0
    for i, (train_id, test_id) in enumerate(kf.split(X,y)):
        model = CNN()
        model.to(device)
        error = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-5)
        X_train = X[train_id]
        y_train = y[train_id]
        X_val = X[test_id]
        y_val = y[test_id]

        training_set = data.TensorDataset(X_train,y_train)
        training_set_loader = data.DataLoader(training_set,batch_size=1000)
        val_set = data.TensorDataset(X_val,y_val)
        val_set_loader = data.DataLoader(val_set,batch_size=1000)

        loss_list = []
        acc_list = []
        count = 0
        acc_list, loss_list, count = f.train(fold_id = i+1,model = model, error = error, optimizer = optimizer, TrainLoader = training_set_loader, ValLoader = val_set_loader)
        
        if (acc_list[-1]>best_val_score):
            best_val_score = acc_list[-1]
            torch.save(model.state_dict(), "model3.pth")
            best_loss_list=loss_list
            best_acc_list = acc_list
        fold_val_score.append(acc_list[-1])
    print(fold_val_score)
    f.plot(best_acc_list,best_loss_list,count)

else:
    model = CNN()
    model.to(device)
    model.load_state_dict(torch.load("model3.pth"))
    """
    test_path = path+"/test/test.png"
    test_img = cv2.imread(test_path,0)
    test_img = cv2.resize(test_img,(28,28))
    test_img = np.array(test_img)

    prediction = f.predict(model,test_img)
    plt.imshow(test_img)
    plt.title("Predict: "+str(prediction))
    plt.show()

    """
    test_path = os.listdir(path+"/data/testSet")
    plt.subplot(2,5,1)
    a = np.random.randint(2000)

    for id, i in enumerate(test_path[a:a+10]):
        img = cv2.imread(os.path.join(path+"/data/testSet",i),0)
        img = cv2.resize(img,(28,28))
        img = np.array(img)
        prediction = f.predict(model,img)
        plt.subplot(2,5,id+1)
        plt.title("Predict: "+str(prediction))
        plt.imshow(img)
    plt.show()
    

