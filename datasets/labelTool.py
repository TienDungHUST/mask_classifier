
import os, cv2

category=[]
link = os.walk("Mask/val1/mask")
for root,folder,file in link:
    for f in file:
        path = root + "/" + f
        category.append(path)
        print(path)

for i in range(len(category)):
    img = cv2.imread(category[i])
    cv2.imshow("img", img)
    cv2.waitKey(300)
    label = input("Sex label: ")
    name = str(category[i].split("/")[-1].split("_")[0])+ "_" + label + "_" + "%04d" %i + ".jpg"
    try :
        if label == "1": 
            path = os.path.join('Mask/val2/nam', name)
        elif label == "0":
            path = os.path.join('Mask/val2/nu', name)
        else: 
            path = os.path.join('Mask/val2/null', name)
    except: 
        pass
    cv2.imwrite(path, img)
    print(i)