#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:00:23 2020

@author: adamwasserman
"""

import os
import numpy as np #images come as numpy arrays; kept to be safe
import cv2
import torch
import torchvision
"""
File set-up: Have the 6 image folders in a single directory
Pass the directory as the first argument to the preprocess function
All the healthy folders should begin with "Healthy"
and all the patient files with "Patient"
The shape drawn should follow the subject's condition
Naming should be in camel-case (no plural!)
EX: HealthySpiral
"""
#The data below represents the largest row and column size for each category
dimensions = {"Meander": (744,822), "Spiral":(756,786),"Circle":(675,720)} # no longer used




def padWithWhite(img):
    row_add = 756 - img.shape[0]
    col_add = 822 - img.shape[1]
    top = row_add//2
    bot = top if row_add % 2 == 0 else top + 1
    left = col_add//2
    right = left if col_add % 2 == 0 else left + 1
    new_img = cv2.copyMakeBorder(img,top,bot,left,right,cv2.BORDER_CONSTANT, value = [255,255,255])
    return new_img

def preprocess(inPath,outPath):
    """Uploads data into a numpy array
        parameter: filePath – the path to the Image_Data folder
        returns: a tuple containing a numpy array of the data and an vertical vector
        with the corresponding values
    """
    
    data = []
    values = [] # 1 for PD and 0 for Healthy
    
    DATADIR = inPath
    cat1 = ["Healthy","Patient"]
    cat2 = ["Meander","Spiral"]
    for health in cat1:
        tag = "H" if health == 'Healthy' else "P"
        size = 38 if health == "Healthy" else 32
        for subject in range(1,size+1):
            for i in range (1,5):
                delete = False
                temp = []
                for shape in cat2:
                    abrev = 'mea' if shape == 'Meander' else 'sp'
            
                    path = os.path.join(DATADIR,health+shape)
                    img_name = abrev + str(i) + '-' + tag+str(subject)+'.jpg'
                    img_array = cv2.imread(os.path.join(path,img_name))
                    
                    if img_array is None: # look for missing data
                        delete = True
                    else: #can only perform if img_array isn't None
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        temp.append(torch.from_numpy(padWithWhite(img_array)))
                        
                path = os.path.join(DATADIR,health+"Circle")
                img_name = "circA-P"+str(subject)+".jpg"
                img_array = cv2.imread(os.path.join(path,img_name),cv2.COLOR_BGR2RGB)
                
                if img_array is None or delete == True: # datapoints with missing data
                    temp.clear()
                    continue
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                temp.append(torch.from_numpy(padWithWhite(img_array)))#hard-coded for now
                data.append(torch.stack(temp))
                values.append(1.0 if health == "Patient" else 0.0)
    
    
    data = torch.stack(data)
    values = torch.tensor(values)
    data = data.type('torch.FloatTensor')
    data /= 255.0
    data = data.permute(0,1,4,2,3)
    shuffle_index = torch.randperm(259)
    
    X,y= data[shuffle_index], values[shuffle_index]

    X_test,X_train = X[:52], X[52:]
    y_test,y_train = y[:52], y[52:]
     
     
    torch.save(X_train,os.path.join(outPath,"X_train.pt"))
    torch.save(X_test,os.path.join(outPath,"X_test.pt"))
    torch.save(y_train,os.path.join(outPath,"y_train.pt"))
    torch.save(y_test,os.path.join(outPath,"y_test.pt"))



def getData(filePath):
    """Returns the X_train,X_test,y_train,y_test in that order"""
    
    X_train = torch.load(os.path.join(filePath,"X_train.pt"))
    X_test = torch.load(os.path.join(filePath,"X_test.pt"))
    y_train = torch.load(os.path.join(filePath,"y_train.pt"))
    y_test = torch.load(os.path.join(filePath,"y_test.pt"))
    return X_train,X_test,y_train,y_test
    
preprocess('/projectnb/riseprac/GroupB','/projectnb/riseprac/GroupB')
