#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 15:00:23 2020

@author: adamwasserman
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

data = []

DATADIR = "/Users/adamwasserman/Documents/Image_Data/"
cat1 = ["Healthy","Patient"]
cat2 = ["Meander","Spiral"]
for healthy in cat1:
    tag = "H" if healthy == 'Healthy' else "P"
    size = 38 if healthy == "Healthy" else 32
    for j in range(1,size+1):
        temp = []
        for category in cat2:
            abrev = 'mea' if category == 'Meander' else 'sp'
            for i in range (1,5):
                path = os.path.join(DATADIR,healthy+category)
                img_name = abrev + str(i) + '-' + tag+str(j)+'.jpg'
                img_array = cv2.imread(os.path.join(path,img_name),cv2.IMREAD_GRAYSCALE)
                #new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                temp.append(img_array)
        path = os.path.join(DATADIR,healthy+"Circle")
        img_name = "circA-P"+str(j)+".jpg"
        img_array = cv2.imread(os.path.join(path,img_name),cv2.IMREAD_GRAYSCALE)
        temp.append(img_array)
        data.append(tuple(temp))
                
np_data = np.array(data)
            