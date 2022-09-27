import os
a = []
#string = "E:\\DATN_EE5020\\Datastet\\RMFD\\AFDB\\self-built-masked-face-recognition-dataset\\"
name = os.walk("/home/wiccan/Documents/DATN/Dataset/masked_raw")
for root,folder,file in name:
    for f in file:
        path = root + "\\" + f
        a.append(path)
       # a.append(path.replace(string,"").replace("\","/"))

f = open("/home/wiccan/Documents/DATN/Dataset/link1.txt","w+")
for i in a:
    string1 = i + "\n"
    f.write(string1)