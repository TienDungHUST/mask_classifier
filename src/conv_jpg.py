
import os
import cv2
import random

l=0
link = []
name = os.walk("/home/wiccan/Documents/DATN/Dataset/Eye_full_data/Open")
for root,folder,file in name:
    for f in file:
        path = root + "/" + f
        link.append(path)
# f = open("/home/wiccan/Documents/DATN/Dataset/link3.txt","w+")
# for i in link:
#     string1 = i + "\n"
#     f.write(string1)
for k in range(40804, len(link)):
	# m = link[k].split(os.path.sep)[-1].split(".")[0]
	# if m == 'm':
		# l=l+1
		img = cv2.imread(link[k])  
		print(k)     
		cv2.imwrite("/home/wiccan/Documents/DATN/Dataset/Eye_full_data/Test/open/open_%d.jpg" %(k), img)
