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

#The data below represents the largest row and column size for each category
dimensions = {"Meander": (744,822), "Spiral":(756,786),"Circle":(675,720)}


def padWithAvg(img, newRow, newCol):
    avg = np.median(img)
    row_add = newRow - img.shape[0]
    col_add = newCol - img.shape[1]
    top = row_add//2
    bot = top if row_add % 2 == 0 else top + 1
    left = col_add//2
    right = left if col_add % 2 == 0 else left + 1
    new_img = cv2.copyMakeBorder(img,top,bot,left,right,cv2.BORDER_CONSTANT, value = avg)
    return new_img



def uploadData(filePath):
    data = []
    
    DATADIR = filePath
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
                    img_array = cv2.imread(os.path.join(path,img_name),cv2.IMREAD_GRAYSCALE)
                    
                    if img_array is None: # look for missing data
                        delete = True
                    else: #can only perform if img_array isn't None
                        newShape = dimensions[category]
                        temp.append(padWithAvg(img_array,*newShape))
                        
                path = os.path.join(DATADIR,healthy+"Circle")
                img_name = "circA-P"+str(j)+".jpg"
                img_array = cv2.imread(os.path.join(path,img_name),cv2.IMREAD_GRAYSCALE)
                
                if img_array is None or delete == True: # datapoints with missing data
                    temp.clear()
                    continue
                
                temp.append(padWithAvg(img_array,675,720))#hard-coded for now
                data.append(tuple(temp))
    
    
    
    return np.array(data)

if __name__ == '__main__':
    np_data = uploadData('/Users/adamwasserman/Documents/Image_Data')
    
    
    
    
    
    
    
    
    