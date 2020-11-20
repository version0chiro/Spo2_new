import os

from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

def detectionFrontFace(frame):
    cv2.imshow("face",frame)
    # frame=imutils.resize(frame,width=300,height=300)
    net = cv2.dnn.readNetFromCaffe("models/deploy.prototxt.txt", "models/res10_300x300_ssd_iter_140000.caffemodel")
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,
                                 (300,300),(104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0,detections.shape[2]):
        confidence = detections[0,0,i,2]
        print(confidence)
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        x=startX-endX
        y=startY-endY
        area= x*y
        # print(area)
        # print(box)
        if confidence < 0.9660 or area>6500:
            # print("nahi mila face")
            return 0
        
        else:
            return 1
    