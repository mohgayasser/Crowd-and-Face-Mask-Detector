from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from retinaface.pre_trained_models import get_model 
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

#Retina
import torch
import tqdm
from imutils import paths
import os
import cv2
import imutils
import time
import albumentations as A
import torch
from tqdm import tqdm


def detect_predict(frame, faceNet, maskNet):
    # Get dimension and create blob

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    
  
    faceNet.eval()
    detections = faceNet.predict_jsons(frame)
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []


    predictions = []
    with torch.no_grad():
      for detections in tqdm(detections):
        x_min, y_min, x_max, y_max = detections['bbox']

        x_min = np.clip(x_min, 0, x_max)
        y_min = np.clip(y_min, 0, y_max)
        # ensure the bounding boxes fall within the dimensions of
        # the frame
        (x_min, y_min) = (max(0, x_min), max(0, y_min))
        (x_max, y_max) = (min(w - 1, x_max), min(h - 1, y_max))

        # extract the face ROI, convert it from BGR to RGB channel
        # ordering, resize it to 224x224, and preprocess it
        face = frame[y_min:y_max, x_min:x_max]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)

        # add the face and bounding boxes to their respective
        # lists
        faces.append(face)
        locs.append((x_min, y_min, x_max, y_max))
         
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

faceNet = get_model("resnet50_2020-07-20", max_size=2048)
maskNet = load_model('/content/drive/MyDrive/mask_detector_NASNetNewData.model')

def maskde():
    frame = cv2.imread("result.png")
    frame = imutils.resize(frame, width=600, height=800)
    (locs, preds) = detect_predict(frame, faceNet, maskNet)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (withoutMask, mask) = pred

        # determine the class label and color we'll use to draw
        label = "No Mask" if withoutMask > mask else "Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        #cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.imwrite("/content/result.png",frame)
        

