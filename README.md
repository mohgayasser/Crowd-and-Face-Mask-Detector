# Crowd-and-Face-Mask-Detector
In this project, a two-stage Face Mask Detector and Crowed counting is implemented.
The First stage Crowed counting  Using SDC-net model for doing crowd counting that is model learned in closed set and can be generalized to open set.
It first generate the ground truth of input image then the density map, the do feature extraction 
by VGG16 model the divide the maps to count each part separately then evaluate the count for all parts then we append this counter to input image. 
The second stage uses a pretrained Retina Face model for face detection. Then training Face Mask Classifier models trying NasNet Mobile and MobileNetV2 on dataset and based on performance, 
the NasNet model was selected for classifying faces as masked or non-masked 

[![Watch the video](https://drive.google.com/file/d/15I9F9vue39prjlkocUjTNDoI2MRbitKp/view)
