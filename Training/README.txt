
This repository contains the code for model training and evaluation on the published datasets ShanghaiTech Part_A and Part_B and other datasets but we will discuss the description of shanghaiTech.
## Environment
Install required packages according to `requirements.txt`.

our model as two models separately( Counting and Mask Detection) to get the best weights of each one,
 because the type of data that each one use to learn how to work will is completely different.
and in end we combine them by using the best weights got from each one.

##Counting Model 

## Datasets
Download the ShanghaiTech dataset using the links from [this repo](https://github.com/desenzhou/ShanghaiTechDataset) 
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


## Ground truth density maps
Generate ground truth density maps by running Pythone file gen_density_maps.py


Files with the names `density_maps_part_{A|B}_{train,test}.npz` will appear in the current directory.

The generated density maps can be visualized and compared to the pre-calculated density maps provided by the [official repo](https://github.com/xhp-hust-2018-2011/S-DCNet) (only for the test sets of ShanghaiTech Part_A and Part_B).
In order to do so, download the archive `Test_Data.zip` using the links in the `Data` section of the README in the official repo.
After unpacking the archive, you will have the following directory structure:

```
./
└── Test_Data/
    ├── SH_partA_Density_map/
    │   ├── test/
    │   │   ├── gtdens/IMG_{1,2,3,...,182}.mat
    │   │   └── images/IMG_{1,2,3,...,182}.jpg
    │   └── rgbstate.mat
    └── SH_partB_Density_map/
        ├── test/
        │   ├── gtdens/IMG_{1,2,3,...,316}.mat
        │   └── images/IMG_{1,2,3,...,316}.jpg
        └── rgbstate.mat
```

Next, run `gen_density_maps.py` wiand change the path of dataset in the code to  the `gtdens` directory:


Directory named `cmp_dmaps_part_{A|B}_test_<some_random_string>` containing pairs of images (named `IMG_<N>_my.png` / `IMG_<N>_xhp.png`) will be created.


## Training
`train.py` is the script for training a model.


Fine-tuning is supported (check the option `train.pretrained_ckpt`).

The logs and checkpoints generated during training are placed to a folder named like `outputs/<launch_date>/<launch_time>`. 
Plots of MAE and MSE vs epoch number can be visualized by `tensorboard`:


tensorboard --logdir outputs/<date>/<time>



## Evaluation
`evaluate.py` is the script for evaluating a checkpoint. Select a checkpoint for epoch `N` and run a command like this:


python evaluate.py \
    dataset=ShanghaiTech_part_B\
    test.trained_ckpt_for_inference=outputs/<date>/<time>/checkpoints/epoch_<N>.pth


You will get an output like this for part_A.
or like this for part_B.

In our training the training time was limited.
so that the error values are higher than that reported in the [original paper](https://arxiv.org/abs/2001.01886) for SS-DCNet C-Counter (MAE = 56.1, MSE = 88.9 for part_A test set; MAE = 6.6, MSE = 10.8 for part_B test set).

##Mask Detection Model


## Datasets
Download the Face Mask Detection dataset using the links from [kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection) 
```
./
└── face-mask-detection/
    ├── annotations/
    │	│
    │   └── maksssksksss{0,1,2,3,...,852}.xml
    │
    └── images/
 	│
        └── maksssksksss0{0,1,2,3,...,852}.png

```

and from [Github](https://github.com/sidgan22/MaskDetection/tree/master/dataset):

```
./
└── MaskDetection/
    │
    └── dataset/
    	│
        ├── mask_on/
 	│
        └── mask_off/
```

## Training
'TrainMaskDetection.py' is the script for training the model , After train saved file NasNet.model in the same path of the project.
