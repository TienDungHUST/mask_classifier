from mtcnn import MTCNN
import cv2
import os

link = []
name = os.walk("/home/wiccan/Downloads/handonFace")
for root,folder,file in name:
    for f in file:
        path = root + "/" + f
        link.append(path)
print(len(link))
f = open("/home/wiccan/Downloads/handcrop/linksun.txt","w+")
for i in link:
    string1 = i + "\n"
    f.write(string1)

detector = MTCNN()

def crop(link_img,i):
    img = cv2.imread(link_img)
    
    detections = detector.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(detections)
    l=0
    for detection in detections:
        y = max (0, detection['box'][1])
        x = max (0, detection['box'][0])
        h = detection['box'][3]
        w = detection['box'][2]
        # cv2.imshow("img", img)
        result = img[y:y+h, x:x+w,:]
        # cv2.imshow("face",result ) # y:y+h, x:x+w,:
        
        if (detections[0]['confidence']>=0.10):
            # cv2.imwrite("/home/wiccan/Documents/DATN/Dataset/crop3/masked_3_%d_%d.jpg" %(i,l), result)
            cv2.imwrite("/home/wiccan/Downloads/handcrop/mask_hand%d%d.jpg" %(i,l), result)
            
        # cv2.waitKey(-1)
        else: 
            print ("cancel")
        print(detections[0]['confidence']) 
        print(str(i) + "_" +str(l))
        l=+1

for k  in range(len(link)): 
    crop(link[k], k)
