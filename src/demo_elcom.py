from retina.retinaface_cov import RetinaFaceCoV
import cv2,sys 
import time
import numpy as np
from utils import crop, pre_frame_face, predict
from backbone import mobilenetv1, base
import argparse

IMG_HEIGHT = 128
IMG_WIDTH = 128
GPU_ID = -1

def main(args): 
    backbone = args.backbone
    pre_weight = args.pre_weight

    if backbone == "mobilenetv1":
        maskmodel = mobilenetv1(IMG_HEIGHT,IMG_HEIGHT)
    if backbone == "base":
        maskmodel = base(IMG_HEIGHT,IMG_HEIGHT)

    maskmodel.load_weights(pre_weight)
    detector = RetinaFaceCoV('src/retina/model-retina/mnet_cov2', 0, GPU_ID, 'net3l')
    
    if(args.imgpath):
        imgpath = args.imgpath
        frame = cv2.imread(imgpath)
        frame =  cv2.resize(frame, (640,360))
        predict(detector, maskmodel, frame, IMG_HEIGHT,IMG_WIDTH)
        
        cv2.imshow('Result',frame)
        cv2.waitKey(-1) 

    elif(args.videopath):
        videopath = args.videopath
        cap = cv2.VideoCapture(videopath)

        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('/home/wiccan/Videos/Out/demovideo2604.mp4',fourcc, 10.0, (320,240))

        while(True):
            # Capture frame-by-fram
            ret, frame  = cap.read()
            frame =  cv2.resize(frame, (640,360))
            predict(detector, maskmodel, frame, IMG_HEIGHT,IMG_WIDTH)
        
            cv2.imshow('Result',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("NOTE: Parse the image path or video path")

    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, 
        help='Choose backbone base or mobilenetv1.', default='base')
    parser.add_argument('--pre_weight', type=str,
        help='Directory of checkpoint.', default='models/cnn/cp-0020.ckpt')
    parser.add_argument('--videopath', type=str,
        help='Path to the video demo.')
    parser.add_argument('--imgpath', type=str,
        help='Path to the image test.')
    
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    print("=== MASK CLASSIFIER ===")
    main(parse_arguments(sys.argv[1:]))
    
