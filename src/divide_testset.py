import cv2
import os
import random

link = []
name = os.walk("/home/wiccan/Documents/DATN/Dataset/face")
for root,folder,file in name:
    for f in file:
        path = root + "/" + f
        link.append(path)
random.shuffle(link)
x= (int)(0.1*len(link))
print(x)
for k in range(0, x):
    img = cv2.imread(link[k])       
    cv2.imwrite("/home/wiccan/Documents/DATN/Dataset/testset/non_mask/test_nonmask%d.jpg" %k, img)
for k in range(x, len(link)):
    img = cv2.imread(link[k])       
    cv2.imwrite("/home/wiccan/Documents/DATN/Dataset/trainset/non_mask/train_nonmask_%d.jpg" %k, img)