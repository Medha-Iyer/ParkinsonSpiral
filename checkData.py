#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:05:18 2020

@author: adamwasserman
"""

import torch



Xm_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xm_test.pt')
Xs_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xs_test.pt')
Xc_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xc_test.pt')
y_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/y_test.pt')
Xm_train = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xm_train.pt')
Xs_train = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xs_train.pt')
Xc_train = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xc_train.pt')
y_train = torch.load('/projectnb/riseprac/GroupB/preprocessedData/y_train.pt')

remove_indices = torch.ones(Xm_train.shape[0])

shape_dict = {"Meanders":(Xm_train,Xm_test),"Spirals":(Xs_train,Xs_test),"Circles": (Xc_train,Xc_test)}

for shape in shape_dict:
    for i in range(shape_dict[shape][0].shape[0]):
        for j in range(shape_dict[shape][1].shape[0]):
            if torch.all(torch.eq(shape_dict[shape][0][i],shape_dict[shape][1][j])):
                remove_indices[i] = 0

Xm_train = Xm_train[(remove_indices != 0).nonzero().squeeze()]
Xs_train = Xs_train[(remove_indices != 0).nonzero().squeeze()]
Xc_train = Xc_train[(remove_indices != 0).nonzero().squeeze()]
y_train = y_train[(remove_indices != 0).nonzero().squeeze()]

torch.save(Xm_train,'/projectnb/riseprac/GroupB/preprocessedData/Xm_train1.pt')
torch.save(Xs_train,'/projectnb/riseprac/GroupB/preprocessedData/Xs_train1.pt')
torch.save(Xc_train,'/projectnb/riseprac/GroupB/preprocessedData/Xc_train1.pt')
torch.save(y_train,'/projectnb/riseprac/GroupB/preprocessedData/y_train1.pt')

print(Xm_train.shape,Xs_test.shape,Xc_test.shape,y_test.shape)