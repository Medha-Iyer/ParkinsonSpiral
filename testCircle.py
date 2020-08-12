#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 22:29:35 2020

@author: adamwasserman
"""

import matplotlib.pyplot as plt
import torch
from spiralArch import SpiralConv
import seaborn as sns
from statistics import mean
import Dataset

X_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xc_test.pt')
y_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/y_test.pt')

dataset = Dataset.Dataset(X_test, y_test)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=13,shuffle=False)

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

for j, (X,y) in enumerate(data_loader):
    if(j>= 4):
        break
    X = X.to(device)
    y = y.to(device)
    NN.forward(X)
    yhat = NN.forward(X).reshape(13)
    yhat = (yhat>0.5).float()
    break
    acc = 0
    for pred,actual in zip(yhat.tolist(),y.tolist()):
            conf_mat[int(actual),int(pred)] += 1
            acc += 1.0 if pred == actual else 0.0
    
    break
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
#plt.savefig('/projectnb/riseprac/GroupB/Images/CircleAccuracyFINAL.png')

plt.plot(x,precision,color='b',label = 'precision')
plt.plot(x,recall,color='r', label = 'recall')
plt.plot(x,f1,color='k',label = 'f1 score')
plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Score (%)")
#plt.savefig('/projectnb/riseprac/GroupB/Images/CircleScoresFINAL.png')

sns_plot = sns.heatmap(conf_mat/torch.sum(conf_mat), annot=True,
            fmt='.2%', cmap='Blues')
conf_img = sns_plot.get_figure()    
#conf_img.savefig('/projectnb/riseprac/GroupB/Images/conf_mat.png')

print('Accuracy =',mean(accs))
print('Final precision =',precision[-1])
print('Final recall =',recall[-1])
print('Final f1 =',f1[-1])