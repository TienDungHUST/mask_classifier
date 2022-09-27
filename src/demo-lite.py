from retinaface_cov import RetinaFaceCoV
import cv2,os 
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from PIL import ImageDraw

#Khoi tao model
IMG_HEIGHT = 128
IMG_WIDTH = 128
GPU_ID = -1
SCALE = 4.5
thres_close = 5
time_all = []
time_detect = []
time_mask = []
time_sung = []
time_eye = []
#Khoi tao model

def Mask_TFL_model():
    base_model  = keras.applications.MobileNet(
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3),
    include_top = False,
    weights     = 'imagenet'
    )
    base_model.trainable = False
    inputs = keras.Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training = False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = keras.layers.Dense(2)(x)
    outputs = layers.Activation('softmax')(x)
    model   = keras.Model(inputs, outputs)
    model.compile(  optimizer = 'adam',
                    loss    = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics =['accuracy'])
    return model
def Sung_crop_base_model(): 
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters = 8,kernel_size = (3,3), strides = 2, padding = 'valid',
                            activation =  'relu', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.MaxPool2D(pool_size = (2,2)))
    model.add(layers.Conv2D(filters = 16,kernel_size = (3,3), strides = 2, padding = 'valid',
                            activation =  'relu'))
    model.add(layers.MaxPool2D(pool_size = (2,2)))
    model.add(layers.Conv2D(filters = 32,kernel_size = (3,3), strides = 2, padding = 'valid',
                            activation =  'relu'))      
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))

    # sgd= tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, name='SGD2')

    model.compile(  optimizer = 'adam', 
                    loss      = tf.keras.losses.CategoricalCrossentropy(),
                    metrics   = 'accuracy')
    return model
def Eyes_CNN03_model():

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters = 16,kernel_size = (3,3), strides = 2, padding = 'valid',
                            activation = 'relu', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)))
    model.add(layers.MaxPool2D(pool_size = (2,2)))
    model.add(layers.Conv2D(filters = 32,kernel_size = (3,3), strides = 2, padding = 'valid',
                            activation =  'relu'))
    model.add(layers.MaxPool2D(pool_size = (2,2))) 
    model.add(layers.Conv2D(filters = 64,kernel_size = (3,3), strides = 2, padding = 'valid',
                            activation = 'relu'))  
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))


    model.compile(  optimizer = 'adam', 
                    loss      = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    metrics   = 'accuracy')
    return model
def Eyes_CNN04_model():

    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters= 16,kernel_size= (3,3), strides= 2, padding= 'valid',
                            activation=  'relu', input_shape= (80, 80, 3)))
    model.add(layers.MaxPool2D(pool_size= (2,2)))
    model.add(layers.Conv2D(filters= 32,kernel_size= (3,3), strides= 2, padding= 'valid',
                            activation=  'relu'))
    model.add(layers.MaxPool2D(pool_size= (2,2))) 
    model.add(layers.Conv2D(filters= 64,kernel_size= (3,3), strides= 1, padding= 'valid',
                            activation=  'relu'))  
    model.add(layers.GlobalMaxPooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))


    model.compile(optimizer='adam', 
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics='accuracy')
    return model
#Cac ham phu tro

def detect(img):
    faces, landmarks = detector.detect(img)
    print(faces)
    return faces,landmarks
def crop(face,landmark, img):
    box = face[0:4].astype(np.int)
    y1  = box[1]
    x1  = box[0]
    y2  = box[3]
    x2  = box[2]

    x = landmark[2][0]
    y = landmark[2][1]
    eyeL = landmark[0]
    eyeR = landmark[1]

    img_crop_face = img[y1:y2, x1:x2,:]
    img_crop_sung = img[y1:y, x1:x2,:]
    start_point = (x1, y1)
    stop_point  = (x2,y2)
    return img_crop_face, img_crop_sung, start_point,stop_point,eyeL, eyeR

def pre_frame_face(image):
    size = (IMG_HEIGHT, IMG_WIDTH)
    img  = cv2.resize(image, size)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("face", img)
    img  = np.array(img) / 255.0
    img  = np.expand_dims(img, axis = 0)
    return img

def pre_frame_sung(image):
    size = (IMG_HEIGHT, IMG_WIDTH)
    img  = cv2.resize(image, size)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("sung", img)
    img  = np.array(img) / 255.0
    img  = np.expand_dims(img, axis = 0)
    return img
def pre_frame_eye(image):
    size = (80, 80)
    img  = cv2.resize(image, size)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img  = np.array(img) / 255.0
    print(img)
    img  = np.expand_dims(img, axis = 0)
    return img

#Khai bao model va nap weight
take_weight_mask = '/home/wiccan/Downloads/SavedWightModel/Mask_weight/TFL01/cp-0030.ckpt'
take_weight_sung = '/home/wiccan/Downloads/SavedWightModel/Sung_weight/sung_crop_base02/cp-0025.ckpt'
take_weight_eyes = '/home/wiccan/Downloads/SavedWightModel/Eyes_weight/CNN04/cp-0048.ckpt'

maskmodel = Mask_TFL_model()
sungmodel = Sung_crop_base_model()
eyesmodel = Eyes_CNN04_model()

maskmodel.load_weights(take_weight_mask)
sungmodel.load_weights(take_weight_sung)
eyesmodel.load_weights(take_weight_eyes)

detector = RetinaFaceCoV('./model/mnet_cov2', 0, GPU_ID, 'net3l')

count_frame  = 0
cap          = cv2.VideoCapture(0)
closed_times = 0

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/home/wiccan/Videos/Out/demovideo2604.mp4',fourcc, 10.0, (320,240))

while(True):
    # Capture frame-by-frame
    tic         = time.time()
    ret, frame  = cap.read()
    # frame =  cv2.resize(frame, (320,240))
    count_frame = count_frame +1
    tic_detect = time.time()
    faces,landmarks = detect(frame)
    timer_dt = time.time()-tic_detect
    time_detect.append(timer_dt)
    # print("Thoi gian chay detect face: ", time.time()-tic1)
    if faces is not None : 
        for i in range(faces.shape[0]):
            face     = faces[i]
            landmark = landmarks[i].astype(np.int)
            img_crop_face, img_crop_sung, start_point,stop_point, eyeL, eyeR= crop(face,landmark, frame)

            tic_mask = time.time()
            result_class = maskmodel(pre_frame_face(img_crop_face))
            time_mask.append(time.time()-tic_mask)
            # print("Thoi gian chay mask model transfer",time.time()-t1)
            result_class = np.asarray(result_class)
            print ("mask result",result_class)

            tic_sung = time.time()
            sung_rst = sungmodel(pre_frame_sung(img_crop_sung))
            time_sung.append(time.time()-tic_sung)
            # print("Thoi gian chay sung model transfer",time.time()-t2)
            sung_rst = np.asarray(sung_rst)
            print(sung_rst)
            
            sz_crop = [stop_point[0]- start_point[0],stop_point[1]- start_point[1]]
            m = sz_crop[1]/SCALE #eye's box-size

            if result_class[0][0]>=0.5: 
                str1 = f"mask: {100 *result_class[0][0]: 5.2f}%"
                cv2.putText(frame, text = str1, org = (start_point[0], start_point[1]-30), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.50, color = (255, 0, 0), thickness = 2)
            else: 
                str1 = f"nonmask: {100 *result_class[0][1]: 5.2f}%"
                cv2.putText(frame, text= str1, org= (start_point[0], start_point[1]-30), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
            if sung_rst[0][0] >= 0.5: 
                str1 = f"nonsung: {100*sung_rst[0][0]: 5.2f}%"
                cv2.putText(frame, 
                            text      = str1, 
                            org       = (start_point[0], start_point[1]-10), 
                            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.50,
                            color     = (255, 0, 0), 
                            thickness = 2)
                elxs = max(0, (eyeL[0]-int(m/2)))      #eye letf -x-  Start
                elys = max(0, (eyeL[1]-int((3*m)/5)))  #eye letf -y-  Start
                elxt = max(1, (eyeL[0]+int(m/2)))      #eye letf -x-  sTop
                elyt = max(1, (eyeL[1]+int((2*m)/5)))  #eye letf -y-  sTop

                erxs = max(0,  (eyeR[0]-int(m/2)))
                erys = max(0,  (eyeR[1]-int((3*m)/5)))
                erxt = max(1, (eyeR[0]+int(m/2)))
                eryt = max(1, (eyeR[1]+int((2*m)/5)))
                print("eye left ", elxs," ", elys," ", elxt," ", elyt)
                print("eye right ", erxs," ", erys," ", erxt," ", eryt)
                print("face crop", start_point, " ", stop_point )

                eye_left = frame[elys:elyt, elxs:elxt, : ]
                tic_eye = time.time()
                eye_left_rst = eyesmodel(pre_frame_eye(eye_left))
                time_eye.append(time.time()-tic_eye)
                # print("Thoi gian chay eye model transfer",time.time()-t3)
                cv2.rectangle(frame, pt1 = ( elxs, elys), pt2 = (elxt, elyt),color = (0, 255, 0), thickness = 1)

                eye_right = frame[erys:eryt, erxs:erxt,: ]
                eye_right_rst = eyesmodel(pre_frame_eye((eye_right)))
                cv2.rectangle(frame, pt1= ( erxs, erys), pt2= (erxt, eryt),color = (0, 255, 0), thickness =1)
                cv2.putText(frame, 
                            text      = f"Closed: LE {100*eye_left_rst[0][0]: 5.2f}% || RE: {100*eye_right_rst[0][0]: 5.2f}%", 
                            org       = (10, 20 +i*20), 
                            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.50,
                            color     = (100, 255, 255), 
                            thickness = 2)
                if(eye_left_rst[0][1] <0.5 or eye_right_rst[0][1]<0.5): 
                        closed_times = closed_times+1
            else: 
                str1 = f"sung: {100*sung_rst[0][1]: 5.2f}%"
                cv2.putText(frame, 
                            text      = str1, 
                            org       = (start_point[0], start_point[1]-10), 
                            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale = 0.50,
                            color     = (255, 0, 0), 
                            thickness = 2)
            frame = cv2.rectangle(frame, pt1 = start_point, pt2 = stop_point, color = (0,225,0), thickness = 1) 
    toc   = time.time()
    timer = toc-tic
    time_all.append(timer)
   

    print("count_frame",count_frame)
    # Canh bao neu nham mat 4 frame /1s
    if (1.0-count_frame*timer >= 0): 
        
        if closed_times >= thres_close: 
            cv2.putText(frame, 
                        text      = f"WARNING !!!", 
                        org       = (300, 50), 
                        fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale = 0.75,
                        color     = (0, 0, 255), 
                        thickness = 2)
            print("closed_times",closed_times)
            
    else: 
        count_frame  = 0
        closed_times = 0
        
    print("THOI GIAN HANDLE 1 FRAME", timer)
    # out.write(frame)
    cv2.imshow('Result',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
f1 = open("/home/wiccan/Documents/DATN/Chart/totaltime.txt","w+")
for i in time_all:
    string1 = f"{i:5.4f}" + ", "
    f1.write(string1)
f2 = open("/home/wiccan/Documents/DATN/Chart/detecttime.txt","w+")
for i in time_detect:
    string1 = f"{i:5.4f}" + ", "
    f2.write(string1)
f3 = open("/home/wiccan/Documents/DATN/Chart/masktime.txt","w+")
for i in time_mask:
    string1 = f"{i:5.4f}" + ", "
    f3.write(string1)
f4 = open("/home/wiccan/Documents/DATN/Chart/sungtime.txt","w+")
for i in time_sung:
    string1 = f"{i:5.4f}" + ", "
    f4.write(string1)
f5 = open("/home/wiccan/Documents/DATN/Chart/eyetime.txt","w+")
for i in time_eye:
    string1 = f"{i:5.4f}" + ", "
    f5.write(string1)

# When everything done, release the capture
cap.release()
# out.release()
cv2.destroyAllWindows()
