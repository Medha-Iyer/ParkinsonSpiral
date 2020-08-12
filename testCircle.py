#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:29:35 2020

@author: adamwasserman
"""

import matplotlib.pyplot as plt
import torch
from circleArch import CircleConv
import seaborn as sns
from statistics import mean

X_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xc_test.pt')
y_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/y_test.pt')

conf_mat = torch.zeros(2,2)
accs = []
precision = []
recall = []
f1 = []

device=torch.device('cuda:0')
NN = CircleConv(num_classes=1,size = (238,211)) #hardcoded for now
NN.load_state_dict(torch.load('/projectnb/riseprac/GroupB/CircleState_dict3.pt'))
NN.eval()
NN.to(device)

for i in range(1,5):
    X = X_test[13*(i-1):13*i].to(device)
    y = y_test[13*(i-1):13*i].to(device)
    NN.forward(X)
    yhat = NN.forward(X).reshape(13)
    yhat = (yhat>0.5).float()
    
    acc = 0
    for pred,actual in zip(yhat.tolist(),y.tolist()):
            conf_mat[int(actual),int(pred)] += 1
            acc += 1.0 if yhat == y else 0.0
    
    accs.append(acc/13)
    l_precision = (conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[0,1]))
    precision.append(l_precision)
    l_recall = (conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[1,0]))
    recall.append(l_recall)
    f1.append(2* ((l_precision*l_recall)/(l_precision+l_recall)))


x = list(range(4))

plt.plot(x,accs,color = 'g')
plt.xlabel('Batches')
plt.ylabel('Accuracy (dec)')
plt.savefig('/projectnb/riseprac/GroupB/Images/CircleAccuracyFINAL.png')

plt.plot(x,precision,color='b',label = 'precision')
plt.plot(x,recall,color='r', label = 'recall')
plt.plot(x,f1,color='k',label = 'f1 score')
plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Score (%)")
plt.savefig('/projectnb/riseprac/GroupB/Images/CircleScoresFINAL.png')
