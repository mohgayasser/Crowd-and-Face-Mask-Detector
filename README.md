# Crowd-and-Face-Mask-Detector
In this project, a two-stage Face Mask Detector and Crowed counting is implemented.
The First stage Crowed counting  Using SDC-net model for doing crowd counting that is model learned in closed set and can be generalized to open set.
It first generate the ground truth of input image then the density map, the do feature extraction 
by VGG16 model the divide the maps to count each part separately then evaluate the count for all parts then we append this counter to input image. 
The second stage uses a pretrained Retina Face model for face detection. Then training Face Mask Classifier models trying NasNet Mobile and MobileNetV2 on dataset and based on performance, 
the NasNet model was selected for classifying faces as masked or non-masked 


![image](https://user-images.githubusercontent.com/44075267/149832134-ee7c48d0-c7b9-4ec7-9408-c27311bddcfe.png)

# User Manual


•	Before Running the Program the user's PC should Contain Pycharm and MATLAB then install all needed packages which are :
1.	tkinter
2.	pillow
3.	cv2
4.	python==3.6.2
5.	pytorch>=0.4.0
6.	numpy==1.14.0
7.	scikit-image==0.13.1
8.	scipy==1.0.0
9.	pandas==0.22.0
10.	h5py
11.	tensorflow 2.5.0
12.	retina-face
•	After the user make sure that all packages are installed successfully, he needs to open the MATLAB program in the Background.
•	After that run the program, the following window will appear as
shown in next Figure 36.

![image](https://user-images.githubusercontent.com/44075267/149832279-9439058c-3910-4fae-9c87-5015e1a23b47.png)
 

•	Uploading Image 
1.	Click on the open image button to browse the images as shown in Figure 37, and select the image that you want to know the number of people on it and detect wether wearing a mask or not and know if the mask was worn correctly or not, then press open.


![image](https://user-images.githubusercontent.com/44075267/149832294-ed83ffef-3b4f-402a-a3b1-1c9ed2fdd449.png)


2.	A Window Contain selected image will be opened as shown in Figure 38, the user just needs to double-click on the middle of this image ,after that it will close by itself after this click.
  
![image](https://user-images.githubusercontent.com/44075267/149832311-b4bc9d12-c7aa-4320-8ede-07342ec7dc7f.png)

3.	After a few seconds the user will find the input image appear on left side of the start window and a result image appear on the right side as shown in Figure 39.

 
![image](https://user-images.githubusercontent.com/44075267/149832330-f97a8894-6b12-4a06-b616-f670675f445c.png)

•	Capturing image:
Click on the Capture Image button as shown in Figure 36, and
then press the (s button) to capture and save the image you want to know the number of people on it and detect if wearing a mask or not and know if the mask was worn correctly or not then repeat steps of Uploading Image Starting From Step 2.



# guide Video

https://user-images.githubusercontent.com/44075267/149837344-a87f6ced-89f0-4252-81a4-1a441b554169.mp4


