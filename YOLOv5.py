#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!python -m pip install --upgrade pip

#!pip install tensorflow==2.3.1 

#!pip install tensorboard==2.4.1
get_ipython().system('pip install torch  #YOLOv5 runs on top of PyTorch, so we need to import it to the notebook')


# In[2]:


import torch # YOLOv5 implemented using pytorch


# In[3]:


from IPython.display import Image #this is to render predictions


# In[4]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[5]:


get_ipython().run_line_magic('cd', 'yolov5')


# In[6]:


get_ipython().system('pip install -r requirements.txt')


# In[21]:


#unzipped data
import glob
import zipfile

for file in files:
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall('data/raw')


# Divide the dataset in train and val folder.

# In[47]:


import os
from random import choice
import shutil
import pathlib


#arrays to store file names
imgs =[]
xmls =[]

#setup dir names
trainPath = '/home/ubuntu/notebooks/Hari_Solomons_Development/az/yolov5/dataset/images/train'
valPath = '/home/ubuntu/notebooks/Hari_Solomons_Development/az/yolov5/dataset/images/val'
crsPath = '/home/ubuntu/notebooks/Hari_Solomons_Development/az/yolov5/data/raw/ts' #dir where images and annotations stored

#setup ratio (val ratio = rest of the files in origin dir after splitting into train and test)
train_ratio = 0.8
val_ratio = 0.2


#total count of imgs
totalImgCount = len(os.listdir(crsPath))/2

#soring files to corresponding arrays
for (dirname, dirs, files) in os.walk(crsPath):
    for filename in files:
        if filename.endswith('.txt'):
            xmls.append(filename)
        else:
            imgs.append(filename)


#counting range for cycles
countForTrain = int(len(imgs)*train_ratio)
countForVal = int(len(imgs)*val_ratio)
print("training images are : ",countForTrain)
print("Validation images are : ",countForVal)


# In[48]:


trainimagePath = '/home/ubuntu/notebooks/Hari_Solomons_Development/az/yolov5/dataset/images/train'
trainlabelPath = '/home/ubuntu/notebooks/Hari_Solomons_Development/az/yolov5/dataset/labels/train'
valimagePath = '/home/ubuntu/notebooks/Hari_Solomons_Development/az/yolov5/dataset/images/val'
vallabelPath = '/home/ubuntu/notebooks/Hari_Solomons_Development/az/yolov5/dataset/labels/val'
#cycle for train dir
for x in range(countForTrain):

    fileJpg = choice(imgs) # get name of random image from origin dir
    fileXml = fileJpg[:-4] +'.txt' # get name of corresponding annotation file

    #move both files into train dir
    #shutil.move(os.path.join(crsPath, fileJpg), os.path.join(trainimagePath, fileJpg))
    #shutil.move(os.path.join(crsPath, fileXml), os.path.join(trainlabelPath, fileXml))

    shutil.copy(os.path.join(crsPath, fileJpg), os.path.join(trainimagePath, fileJpg))
    shutil.copy(os.path.join(crsPath, fileXml), os.path.join(trainlabelPath, fileXml))


    #remove files from arrays
    imgs.remove(fileJpg)
    xmls.remove(fileXml)



#cycle for test dir   
for x in range(countForVal):

    fileJpg = choice(imgs) # get name of random image from origin dir
    fileXml = fileJpg[:-4] +'.txt' # get name of corresponding annotation file

    #move both files into train dir
    #shutil.move(os.path.join(crsPath, fileJpg), os.path.join(valimagePath, fileJpg))
    #shutil.move(os.path.join(crsPath, fileXml), os.path.join(vallabelPath, fileXml))
    shutil.copy(os.path.join(crsPath, fileJpg), os.path.join(valimagePath, fileJpg))
    shutil.copy(os.path.join(crsPath, fileXml), os.path.join(vallabelPath, fileXml))
    
    #remove files from arrays
    imgs.remove(fileJpg)
    xmls.remove(fileXml)

#rest of files will be validation files, so rename origin dir to val dir
#os.rename(crsPath, valPath)
shutil.move(crsPath, valPath)


# running yolov5

# In[58]:


get_ipython().system('python train.py --img 415 --batch 16 --epochs 30 --data dataset.yaml --weights yolov5s.pt --cache')


# In[60]:



get_ipython().system('python detect.py --source runs/train/exp7/a.jpg --weights runs/train/exp7/weights/best.pt')


# In[61]:


get_ipython().system('python detect.py --source runs/train/exp7/b.jpg --weights runs/train/exp7/weights/best.pt')


# In[ ]:




