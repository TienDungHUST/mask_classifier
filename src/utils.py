import cv2
import numpy as np


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

def pre_frame_face(image,IMG_HEIGHT,IMG_WIDTH ):
    size = (IMG_HEIGHT, IMG_WIDTH)
    img  = cv2.resize(image, size)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("face", img)
    img  = np.array(img) / 255.0
    img  = np.expand_dims(img, axis = 0)
    return img
def predict(detector, maskmodel, frame, IMG_HEIGHT,IMG_WIDTH):

    faces,landmarks = detector.detect(frame)
    if faces is not None : 
            for i in range(faces.shape[0]):
                face     = faces[i]
                landmark = landmarks[i].astype(np.int)
                img_crop_face, _, start_point,stop_point, _, _= crop(face,landmark, frame)

                result_class = maskmodel(pre_frame_face(img_crop_face,IMG_HEIGHT,IMG_WIDTH))
                result_class = np.asarray(result_class)
            
                if result_class[0][0]>=0.5: 
                    str1 = f"mask: {100 *result_class[0][0]: 5.2f}%"
                    cv2.putText(frame, text = str1, org = (start_point[0], start_point[1]-30), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale = 0.50, color = (0, 0, 255), thickness = 2)
                else: 
                    str1 = f"nonmask: {100 *result_class[0][1]: 5.2f}%"
                    cv2.putText(frame, text= str1, org= (start_point[0], start_point[1]-30), fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.50, color=(255, 0, 0), thickness=2)
                
                frame = cv2.rectangle(frame, pt1 = start_point, pt2 = stop_point, color = (0,225,0), thickness = 1) 
    return frame