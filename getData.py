#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:00:23 2020

@author: adamwasserman
"""

import os
import numpy as np
import cv2
#import torch
#import torchvision


#The data below represents the largest row and column size for each category
dimensions = {"Meander": (744,822), "Spiral":(756,786),"Circle":(675,720)}


def padWithWhite(img, newRow, newCol):
    row_add = newRow - img.shape[0]
    col_add = newCol - img.shape[1]
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
    for healthy in cat1:
        tag = "H" if healthy == 'Healthy' else "P"
        size = 38 if healthy == "Healthy" else 32
        for j in range(1,size+1):
            for i in range (1,5):
                delete = False
                temp = []
                for category in cat2:
                    abrev = 'mea' if category == 'Meander' else 'sp'
            
                    path = os.path.join(DATADIR,healthy+category)
                    img_name = abrev + str(i) + '-' + tag+str(j)+'.jpg'
                    img_array = cv2.imread(os.path.join(path,img_name))
                    
                    if img_array is None: # look for missing data
                        delete = True
                    else: #can only perform if img_array isn't None
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                        newShape = dimensions[category]
                        temp.append(padWithWhite(img_array,*newShape))
                        
                path = os.path.join(DATADIR,healthy+"Circle")
                img_name = "circA-P"+str(j)+".jpg"
                img_array = cv2.imread(os.path.join(path,img_name),cv2.COLOR_BGR2RGB)
                
                if img_array is None or delete == True: # datapoints with missing data
                    temp.clear()
                    continue
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                temp.append(padWithWhite(img_array,675,720))#hard-coded for now
                data.append(temp)
                values.append(1 if healthy == "Patient" else 0)
    
    #trans=torchvision.transforms.Normalize(0.5,0.5,inplace=True)
    for row in range(len(data)):
        for col in range(len(data[0])):
# =============================================================================
#             data[row][col] = torch.from_numpy(data[row][col])
#             data[row][col] = data[row][col].float()
#             data[row][col] = data[row][col].permute(2,0,1)
#             data[row][col] = trans(data[row][col])
# =============================================================================
            data[row][col] = data[row][col].astype('f')
            #data[row][col] = data[row][col].transpose(2,0,1)
            data[row][col] = data[row][col] / 255.0
    shuffle_index = np.random.permutation(263)
    X = np.array(data)
    y = np.array(values)
    X,y = X[shuffle_index],y[shuffle_index]
    X_test,X_train = X[:53], X[53:]
    y_test,y_train = y[:53], y[53:]
    
    
    np.save(os.path.join(outPath,"X_train.npy"),X_train)
    np.save(os.path.join(outPath,"X_test.npy"),X_test)
    np.save(os.path.join(outPath,"y_train.npy"),y_train)
    np.save(os.path.join(outPath,"y_test.npy"),y_test)


def getData(filePath):
    """Returns the X_train,X_test,y_train,y_test in that order"""
    
    X_train = np.load(os.path.join(filePath,"X_train.npy"),allow_pickle=True)
    X_test = np.load(os.path.join(filePath,"X_test.npy"),allow_pickle=True)
    y_train = np.load(os.path.join(filePath,"y_train.npy"),allow_pickle=True)
    y_test = np.load(os.path.join(filePath,"y_test.npy"),allow_pickle=True)
    return X_train,X_test,y_train,y_test
