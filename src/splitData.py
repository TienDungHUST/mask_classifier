import cv2, os
import numpy as np

link = []
name = os.walk('/home/wiccan/Documents/DATN/Dataset/datafullimprove/non_mask')
for root,folder,file in name:
    for f in file:
        path = root + "/" + f
        link.append(path)

np.random.shuffle(link)        
for i in range(len(link)): 
    image = cv2.imread(link[i])
    name_label = link[i].split(os.path.sep)[-1]
    fnametrain = '/home/wiccan/Documents/DATN/Dataset/split/train/non_mask/'
    fnameval = '/home/wiccan/Documents/DATN/Dataset/split/val/non_mask/'
    print(i)
    if i < 0.9*len(link): 
      cv2.imwrite(fnametrain + name_label, image)
    else: 
      cv2.imwrite(fnameval + name_label, image)
