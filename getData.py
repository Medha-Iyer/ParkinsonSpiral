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
#dimensions = {"Meander": (744,822), "Spiral":(756,786),"Circle":(675,720)} # no longer used


dim= {"Meanders": (561,580), "Spiral" : (678,686), "Circle" : (238,211)}


#def preprocess(inPath,outPath):
"""Uploads data into a numpy array
    parameter: filePath – the path to the Image_Data folder
    returns: a tuple containing a numpy array of the data and an vertical vector
    with the corresponding values
"""
outPath = '/projectnb/riseprac/GroupB/processedData'

meanders = []
spirals = []
circles = []
values = [] # 1 for PD and 0 for Healthy

DATADIR = '/Users/adamwasserman/Documents/Image_Data'
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
                    temp.append(torch.from_numpy(cv2.resize(img_array,dim[shape])))
                    
            path = os.path.join(DATADIR,health+"Circle")
            img_name = "circA-P"+str(subject)+".jpg"
            img_array = cv2.imread(os.path.join(path,img_name),cv2.COLOR_BGR2RGB)
            
            if img_array is None or delete == True: # datapoints with missing data
                temp.clear()
                continue
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            temp.append(torch.from_numpy(cv2.resize(img_array,dim["Circle"])))#hard-coded for now
            meanders.append(temp[0])
            spirals.append(temp[1])
            circles.append(temp[2])
            values.append(1.0 if health == "Patient" else 0.0)


meanders,spirals,circles = torch.stack(meanders).type('torch.FloatTensor'),torch.stack(spirals).type('torch.FloatTensor'),torch.stack(circles).type('torch.FloatTensor')
#NxRxCOLxC
values = torch.tensor(values)
meanders /= 255.0
spirals /= 255.0
circles /= 255.0
meanders,spirals,circles = meanders.permute(0,3,1,2), spirals.permute(0,3,1,2), circles.permute(0,3,1,2)
shuffle_index = torch.randperm(259)

Xm, Xs, Xc= meanders[shuffle_index], spirals[shuffle_index], circles[shuffle_index]
y = values[shuffle_index]


Xm_train,Xs_train,Xc_train = Xm[52:], Xs[52:], Xc[52:]
Xm_test,Xs_test,Xc_test = Xm[:52], Xs[:52], Xc[:52]


y_test,y_train = y[:52], y[52:]
 
 
torch.save(Xm_train,os.path.join(outPath,"Xm_train.pt"))
torch.save(Xs_train,os.path.join(outPath,"Xs_train.pt"))
torch.save(Xc_train,os.path.join(outPath,"Xc_train.pt"))
torch.save(Xm_test,os.path.join(outPath,"Xm_test.pt"))
torch.save(Xs_test,os.path.join(outPath,"Xs_test.pt"))
torch.save(Xc_test,os.path.join(outPath,"Xc_test.pt"))
torch.save(y_train,os.path.join(outPath,"y_train.pt"))
torch.save(y_test,os.path.join(outPath,"y_test.pt"))



def getData(filePath):
    """Returns the X_train,X_test,y_train,y_test in that order"""
    
    X_train = torch.load(os.path.join(filePath,"X_train.pt"))
    X_test = torch.load(os.path.join(filePath,"X_test.pt"))
    y_train = torch.load(os.path.join(filePath,"y_train.pt"))
    y_test = torch.load(os.path.join(filePath,"y_test.pt"))
    return X_train,X_test,y_train,y_test
    
#preprocess('/projectnb/riseprac/GroupB','/projectnb/riseprac/GroupB')
