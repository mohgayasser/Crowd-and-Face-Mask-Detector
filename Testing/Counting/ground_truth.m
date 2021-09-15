function [ output_args ] = groungT(args)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

clear all
clc
close all
filePattern = fullfile('C:\Users\user\PycharmProjects\GpGUI\Testing\data\images', '/*.jpg');
ImageFiles = dir(filePattern);
n = length(ImageFiles);
read_path = 'C:\Users\user\PycharmProjects\GpGUI\Testing\data\images\';
store_path = 'C:\Users\user\PycharmProjects\GpGUI\Testing\data\gtdens\';
t = 0;                          %number of files initially in training set

for i=1:n
    im = imread([read_path 'IMG_' num2str(i+t) '.jpg']);
    im = imresize(im, [768 1024]);
    imwrite(im,[read_path 'IMG_' num2str(i+t) '.jpg'], 'jpg'); 
    figure
    imshow(im)
    [x,y] = getpts;
    image_info{1,1}.location = [x y];
    image_info{1,1}.number = size(x,1);
    save([store_path 'GT_IMG_' num2str(t+i) '.mat'], 'image_info')
    close
end
end 