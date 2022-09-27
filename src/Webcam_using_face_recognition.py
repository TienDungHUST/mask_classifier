# Demo Webcam
from mtcnn import MTCNN
import face_recognition
import cv2,os 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import ImageDraw
#Khoi tao model
img_height = 112
img_width = 112
batch_size = 32

def create_model():
    model = tf.keras.Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(2)),
    model.add(layers.Activation('softmax'))


    model.compile(optimizer='adam', 
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics='accuracy')
    return model
def create_cnnmodel():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters= 16,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu', input_shape= (img_height, img_width, 3)))
    model.add(layers.MaxPool2D(pool_size= (2,2)))
    model.add(layers.Conv2D(filters= 32,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu'))
    model.add(layers.Conv2D(filters= 32,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu'))
    model.add(layers.MaxPool2D(pool_size= (2,2))) 
    model.add(layers.Conv2D(filters= 64,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu'))
    model.add(layers.Conv2D(filters= 64,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu'))
    model.add(layers.MaxPool2D(pool_size= (2,2)))     
    model.add(layers.GlobalMaxPool2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))


    model.compile(optimizer='adam', 
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics='accuracy')
    return model
def create_cnn_ip_gray_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(filters= 16,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu', input_shape= (img_height, img_width, 1)))
    model.add(layers.MaxPool2D(pool_size= (2,2)))
    model.add(layers.Conv2D(filters= 32,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu'))
    model.add(layers.Conv2D(filters= 32,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu'))
    model.add(layers.MaxPool2D(pool_size= (2,2))) 
    model.add(layers.Conv2D(filters= 64,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu'))
    model.add(layers.Conv2D(filters= 64,kernel_size= (3,3), strides= 1, padding= 'same',
                            activation=  'relu'))
    model.add(layers.MaxPool2D(pool_size= (2,2)))     
    model.add(layers.GlobalMaxPool2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))


    model.compile(optimizer='adam', 
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics='accuracy')
    return model



def detect(img):
    face_locations = face_recognition.face_locations(img)
    print(face_locations)
    return face_locations
def crop(detections, img):
    # for detection in detections:
    y1 = max (0, detection[0])
    x1 = max (0, detection[1])
    y2 = detection[2]
    x2 = detection[3]
    # cv2.imshow("img", img)
    img_crop = img[y1:y2, x2:x1,:]
    start_point= (x2, y1)
    stop_point = (x1,y2)
    return img_crop, start_point,stop_point
    

def prd(image):
    size = (img_height, img_width)
    img = cv2.resize(image, size)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = img.reshape(img_height, img_width, 1)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis = 0)
    return img

cap = cv2.VideoCapture(0)

#Load trained_weight 
take_weight = '/home/wiccan/Downloads/SavedWightModel/weightCNN2/cp-0028.ckpt'
newmodel = create_cnnmodel()
newmodel.load_weights(take_weight)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    detection_face = detect(frame)
    if detection_face: 
        for detection in detection_face:
            img_crop, start_point,stop_point= crop(detection, frame)
            result_class = newmodel(prd(img_crop))
            result_class = np.asarray(result_class)
            print (result_class)
    
            if result_class[0][0]>=0.5: 
                str1 = f"mask: {result_class[0][0]}"
            else: str1 = f"non-mask: {result_class[0][1]}"
            # text=f'mask: {result_class[0][0]}, non-mask: {result_class[0][1]} '
            cv2.putText(frame, text= str1, org= (start_point[0], start_point[1]-10), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
            # Display the resulting frame
            frame = cv2.rectangle(frame, pt1 = start_point, pt2= stop_point, color=(0,225,0), thickness =1) 
    cv2.imshow('Result',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()