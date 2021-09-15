import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import os
import glob
import cv2

import xml.etree.ElementTree as et


LR = 1e-4
BS = 32
EP = 30
labels = []
data = []

dic = {"image": [], "Dimensions": []}
for i in range(1, 116):
    dic[f'Object {i}'] = []

# Read dataset
for file in os.listdir(
        "/face-mask-detection/annotations"):
    row = []
    xml = et.parse(
        "/face-mask-detection/annotations/" + file)
    root = xml.getroot()
    img = root[1].text
    row.append(img)
    h, w = root[2][0].text, root[2][1].text
    row.append([h, w])

    for i in range(4, len(root)):
        temp = []
        temp.append(root[i][0].text)
        for point in root[i][5]:
            temp.append(point.text)
        row.append(temp)
    for i in range(len(row), 119):
        row.append(0)
    for i, each in enumerate(dic):
        dic[each].append(row[i])
df = pd.DataFrame(dic)

image_directories = sorted(glob.glob(
    os.path.join("/face-mask-detection/images",
                 "*.png")))
classes = ["without_mask", "mask_weared_incorrect", "with_mask"]
for idx, image in enumerate(image_directories):
    img = cv2.imread(image)
    # scale to dimension
    cv2.resize(img, (224, 224))
    # find the face in each object
    indexRowdf = -1
    for indexx, roww in df.iterrows():
        if roww['image'] in image:
            indexRowdf = indexx

    for obj in df.columns[2:]:
        info = df[obj][indexRowdf]
        if info != 0:
            label = info[0]
            info[0] = info[0].replace(str(label), str(classes.index(label)))
            info = [int(each) for each in info]
            face = img[info[2]:info[4], info[1]:info[3]]
            if ((info[3] - info[1]) > 40 and (info[4] - info[2]) > 40):
                try:
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    if (label == "without_mask" or label == "mask_weared_incorrect"):
                        labels.append("mask_off")
                        data.append(face)
                    elif (label == "with_mask"):
                        labels.append("mask_on")
                        data.append(face)

                except:
                    pass

### Read another dataset with last dataset
DIR = r"/MaskDetection/dataset"
CAT = ['mask_off', 'mask_on']

print("Loading Images")
for cat in CAT:
    path = os.path.join(DIR, cat)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))

        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(cat)

print("Images loaded")

## One hot encoding labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

#Split data for 80% train , 20% test
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.20,
                                                  stratify=labels,
                                                  random_state=42)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto',
    baseline=None, restore_best_weights=False
)

# NASNetMobile Model
baseModel = NASNetMobile(weights="imagenet",
                         include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))

myModel = baseModel.output

myModel = AveragePooling2D(pool_size=(7, 7))(myModel)
myModel = Flatten(name="flatten")(myModel)
myModel = Dense(128, activation="relu")(myModel)

myModel = Dense(128, activation="relu")(myModel)

myModel = Dropout(0.5)(myModel)

myModel = Dense(2, activation="softmax")(myModel)

model = Model(inputs=baseModel.input, outputs=myModel)

for layer in baseModel.layers:
    layer.trainable = False

model.compile(loss="binary_crossentropy",
              metrics=["accuracy"])
hist = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EP, )

# Eval
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# serialize the model to disk
model.save("/NasNet.model",
           save_format="h5")

from sklearn.metrics import accuracy_score

accuracy_score(testY.argmax(axis=1), predIdxs)

