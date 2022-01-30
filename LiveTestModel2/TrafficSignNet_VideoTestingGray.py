# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 19:06:25 2022

@author: devji
"""

from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import imutils
import random
import cv2
import os

model_path = "../Train_Model2/trafficsignnet.model"
classified_image_path = r"classified_images"
video_in = "4.mp4"
# load the traffic sign recognizer model
print("[INFO] loading model...")
model = load_model(model_path)
# load the label names
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]
# grab the paths to the input images, shuffle them, and grab a sample
## Prediction
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
cap = cv2.VideoCapture(video_in)
tfps = 30
fps = round(cap.get(cv2.CAP_PROP_FPS))
hop = round(fps/tfps)
curr_frame = 0
i=0
while(True):
    ret, frame = cap.read()
    if not ret:
        break
    if curr_frame % hop == 0:
        data = []
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #RGB
        image = np.array(image)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1) #clahe
        # image = clahe.apply(image) 
        # preprocess the image by scaling it to the range [0, 1]
        image = cv2.merge([image,image,image]) #How to create 3 channel from one channel
        image = image.astype("float32") / 255.0
        #print('aaaaaaaaaaaaaaaaaaaaaa', image.shape)
        image = image[np.newaxis,:,:] #dimension adjustment
       # image = np.expand_dims(image, 2)
        	# make predictions using the traffic sign recognizer CNN
        preds = model.predict(image)
        j = preds.argmax(axis=1)[0]
        label = labelNames[j]
        	# load the image using OpenCV, resize it, and draw the label
        	# on it
        image = frame
        image = imutils.resize(image, width=128)
        cv2.putText(image, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
        		0.45, (0, 0, 255), 2)
        	# save the image to disk
        p = str(classified_image_path+str('/') + str(i) + '.png')
        print(p)
        i+=1
        cv2.imwrite(p, image)
 