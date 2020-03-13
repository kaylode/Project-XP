import cv2
import numpy as np 
import matplotlib.pyplot as plt
import torch

def preprocess_image(img):
    IMG_SIZE = 200
    w,h,c = img.shape
    
    d = max(h,w)/IMG_SIZE*1.0
    
    if h > IMG_SIZE:
        h = int(h/d)
    if w > IMG_SIZE:
        w = int(w/d)
    img_resized = cv2.resize(img,(h,w))

    horizontal = IMG_SIZE - h
    vertical = IMG_SIZE - w

    left = right = int(horizontal/2)
    top = bot = int(vertical/2)

    img_padding = cv2.copyMakeBorder(img_resized,top,bot,left,right,cv2.BORDER_CONSTANT)
    print(img_padding.shape)
    return img_padding
