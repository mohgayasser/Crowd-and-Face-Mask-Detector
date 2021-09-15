we Test our model as two models separately( Counting and Mask Detection) to get the best weights of each one because the type of data that each one use to learn how to work will is completely different.
and in end we combine them by using the best weights got from each one.
 
## Counting Model
This repository contains the code for model training and evaluation on the published datasets ShanghaiTech Part_A and Part_B and other datasets but we will discuss the description of shanghaiTech For Counting part .

## Environment
Install required packages according to `requirements.txt`.

## Datasets

Download the ShanghaiTech dataset For counting part by using the links from [this repo](https://github.com/desenzhou/ShanghaiTechDataset) 
or from [kaggle](https://www.kaggle.com/tthien/shanghaitech). After unpacking the archive, you will have the following directory structure:

```
./
└── ShanghaiTech/
    ├── part_A/
    │   ├── test_data/
    │   │   ├── ground-truth/GT_IMG_{1,2,3,...,182}.mat
    │   │   └── images/IMG_{1,2,3,...,182}.jpg
    │   └── train_data/
    │       ├── ground-truth/GT_IMG_{1,2,3,...,300}.mat
    │       └── images/IMG_{1,2,3,...,300}.jpg
    └── part_B/
        ├── test_data/
        │   ├── ground-truth/GT_IMG_{1,2,3,...,316}.mat
        │   └── images/IMG_{1,2,3,...,316}.jpg
        └── train_data/
            ├── ground-truth/GT_IMG_{1,2,3,...,400}.mat
            └── images/IMG_{1,2,3,...,400}.jpg
```



##Preprocessing (Ground truth density maps)
In the First, you need to run DM.py file to generate Ground-truth and Density map for you database.

the Ground Truth Function  Will be calling  which will call by its turn a Matlab function,so you need first to open your Matlab program in the background.
for each image in your dataset that has no ground-truth, you will need to annotate each head in the image by click only once on the head and when you annotate the last head in the image you will need to double click instead of one click from the opened window that will be opened after calling ground-truth function.
then the opened window will be closed by itself After your double Click.
the annotated heads that you will select in each image will represent the number of people in the  image that later will be used as a target count for our model.

After generating ground-truth successfully the code of the generate density map will be executed by itself.


##Counting.py
After that you will need to run 'counting.py' 

##Mask Detection Model
## Environment
Install required packages according to `requirements.txt`.

## testing 
you should first run 'TrainMaskDetection.py' 

Project should have 'DetectMask.py' 

then you can run 'Run.py'and choose image from disk

'DetectMask.py' tested by 'Nasnet.model' that saved in train phase.