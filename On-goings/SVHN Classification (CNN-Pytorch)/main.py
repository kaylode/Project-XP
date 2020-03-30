import torch
import cv2
import numpy as np 
import matplotlib.pyplot as plt


def sliding_windows(img):
    img = cv2.resize(img,(300,300))
    x,y,_ = img.shape

    size = [50, 30, 10]
    for sz in size:
        stride = int(sz/2)
        for b in range(0,x,stride):
            for a in range(0,y,stride):
                roi = img[b:b+sz,a:a+sz]
                yield roi



img = cv2.imread("test.jpg",1)

sliced_img = sliding_windows(img)
for id, im in enumerate(sliced_img):
    cv2.imwrite("test_imgs/"+str(id)+".jpg",im)

"""cv2.imshow("img",img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""