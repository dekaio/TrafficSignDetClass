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
import time
import sys
#function to predict label for each input box
def predict_label(model, roi, labelNames):
    try:
        image = cv2.resize(roi,(32, 32))
    except:
        return "0"
    image = image[np.newaxis,:,:,np.newaxis]
    image = np.array(image).astype("float32") / 255.0
    
    # make predictions
    preds = model.predict(image)
    if np.max(preds)>0.75:
        j = preds.argmax(axis=1)[0]
        label = labelNames[j]
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", label)
    else:
        label = "other"
    
    return label
    
#function to draw boxes and labels on the first input image
def detected_image(image_path, bbs, labels):
    # image = io.imread(image_path)
    image = image_path
    for box,label in zip(bbs,labels):
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), [172 , 10, 127], 2)
        cv2.putText(image, label, (int(box[0] + box[2]), int(box[1] + box[3])), cv2.FONT_HERSHEY_DUPLEX,0.75, (0, 255, 255), 2)
        
    return image
    
#function to crop box
def crop_roi(image,box):
    h, w = image.shape[:2]
    x_center, y_center = (box[0] * w), (box[1] * h)
    box_width, box_height = (box[2] * w), (box[3] * h)
    x_min, y_min = (x_center - box_width/2), (y_center - box_height/2)
    roi = image[int(y_min):int(y_min+box_height), int(x_min):int(x_min+box_width)]
    return roi, [x_min, y_min,box_width, box_height]

def signal_handler(sig, frame):
    print ("avg fps: ", (time.time() - total_time)/curr_frame/skip_frame)
    print('You pressed Ctrl+C!')
    sys.exit(0)



""" 
    Displays detected and classified image from a video frame
"""
if __name__ == '__main__':

    video_path = "4.mp4"
    output_path = "classified_images"
  #  model_name = "../Train_Model2/trafficsignnet.model/20220116-202944"  # grayscale clahe
    model_name = "trafficsignnet.model/20220117-180435.hdf" 
    cap = cv2.VideoCapture(video_path)
    curr_frame = 0
    skip_frame = 2

    model = load_model(model_name)

    #yolo setup
    net = cv2.dnn.readNet ("../Train_Yolo/yolov4-tiny_training_last.weights", "../Train_Yolo/yolov4-tiny_training.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    confidence_threshold = 0.5
    
    # load the label names
    labelNames = open("../Train_CNNmodel/signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
    total_time = time.time()
    
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        elif np.count_nonzero(np.array(frame.shape))<3:
            continue
        curr_frame+=1
        if curr_frame%skip_frame == 0:
            try:
                start_time = time.time()
                # curr_frame +=1
                # image_path = 'captured_frames/frame'+str(curr_frame)+'.jpg'
                # cv2.imwrite(image_path,frame)
                #         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # print ("Set 3: ", str(round(time.time(), 2)))
                confidences = []
                boxes = []
                rois = []
                labels = []
                bbs=[]


                #forward pass yolo
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                #reading again since classfier performing better on skimage
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                image = clahe.apply(image) 
                # image=cv2.imwrite(image_path, image)
              
                for out in outs:
                    for detection in out:
                        confidence = np.max(detection[5:])

                        if confidence > confidence_threshold:
                            roi, box = crop_roi(image,detection)
                            confidences.append(float(confidence))
                            rois.append(roi)
                            boxes.append(box)

                #adjust overlaps
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                label = ""
                for i,roi in enumerate(rois):
                    if i in indexes:
                        label = predict_label(model, roi, labelNames)
                        labels.append(label)
                        bbs.append(boxes[i])
                if (labels is not None):
                    image = detected_image(frame, bbs, labels)    
                    cv2.imwrite(output_path+"/frame"+str(curr_frame)+'.jpg',image)
                elapsed_time = time.time() - start_time
                fps = 1/elapsed_time
                print ("frame", curr_frame)
                print ("fps: ", str(round(fps, 2)))
            except ValueError:
                continue
            except KeyboardInterrupt:
                print ("avg fps: ", (curr_frame/skip_frame)/(time.time() - total_time))
                cap.release()
                cv2.destroyAllWindows()
                sys.exit(0)
    print ("avg fps: ", (curr_frame/skip_frame)/(time.time() - total_time))
    cap.release()
    cv2.destroyAllWindows()