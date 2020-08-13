#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:29:35 2020

@author: adamwasserman
"""

import matplotlib.pyplot as plt
import torch
from architecture import SimpleConv
import seaborn as sns
from statistics import mean
from CombDataset import Dataset
import numpy as np

Xm_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xm_test1.pt')
Xs_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xs_test1.pt')
Xc_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xc_test1.pt')
y_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/y_test1.pt')

dataset = Dataset(Xm_test,Xs_test,Xc_test, y_test)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=13,shuffle=False)

conf_mat = torch.zeros(2,2)
accs = []
precision = []
recall = []
f1 = []
acc = 0.0

device=torch.device('cuda:0')
NN = SimpleConv(num_classes=1,size = (238,211)) #hardcoded for now
NN.load_state_dict(torch.load('/projectnb/riseprac/GroupB/MAINstate_dict1.pt'))
NN.eval()
NN.to(device)

for j, (Xm,Xs,Xc,y) in enumerate(data_loader):
    batch_size = Xm.shape[0]
    Xm,Xs,Xc = Xm.to(device),Xs.to(device),Xc.to(device)
    y = y.to(device)
    yhat = NN.forward(Xm,Xs,Xc).reshape(batch_size)
    yhat = (yhat>0.5).float()
    
    for pred,actual in zip(yhat.tolist(),y.tolist()):
            conf_mat[int(actual),int(pred)] += 1
            acc += 1.0 if pred == actual else 0.0
    
    
    accs.append(acc/batch_size)
    l_precision = (conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[0,1]))
    precision.append(l_precision)
    l_recall = (conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[1,0]))
    recall.append(l_recall)
    f1.append(2* ((l_precision*l_recall)/(l_precision+l_recall)))
    

fig = plt.figure()

labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
x_axis_labels = ['Healthy', 'PD']
y_axis_labels = ['PD', 'Healthy']
sns_plot = sns.heatmap(conf_mat/torch.sum(conf_mat), annot=labels, fmt='.2', xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap='Blues')
plt.xlabels('Predicted Category')
plt.ylabels('True Category')
conf_img = sns_plot.get_figure()    
conf_img.savefig('/projectnb/riseprac/GroupB/Images/CombConf_matFINAL.png') 

print('Accuracy =',acc/44)
print('Final precision =',precision[-1])
print('Final recall =',recall[-1])
print('Final f1 =',f1[-1])