#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 22:31:40 2020

@author: adamwasserman
"""

from circleArch import CircleConv
import Dataset
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os
from statistics import mean
import seaborn as sns

def evaluate():
    NN.eval()
    test_acc = []
    for j, (Xc,y) in enumerate(test_loader):
        batch_size = Xc.shape[0]
        Xc = Xc.to(device)
        y = y.to(device)
        yhat = NN.forward(Xc).reshape(batch_size)
        yhat = (yhat>0.5).float()
        
        acc = 0.0
        for pred,actual in zip(yhat.tolist(),y.tolist()):
                acc += 1.0 if pred == actual else 0.0
        test_acc.append(acc)
    test_accs.append(sum(test_acc)/52) #there are 52 test datapoints
    print("Test accuracy of=",test_accs[-1])
    if len(test_accs) > 1 and test_accs[-1] - test_accs[-2] < -0.01:
        return breakout + 1
    elif len(test_accs) > 1 and test_accs[-1] - test_accs[-2] < 0:
        return breakout
    else:
        return 0

epochs =  1000#remember for circles it's practically multiplied by 4
batch_size = 10
threshold = 0.5
run_num = 7
losses = []
accs = []
test_accs = []
precision = []
recall = []
f1 = []
breakout = 0

conf_mat = torch.zeros(2,2)

filePath = '/projectnb/riseprac/GroupB/preprocessedData'

X_train = torch.load(os.path.join(filePath,"Xc_train1.pt"))
X_test = torch.load(os.path.join(filePath,"Xc_test.pt"))
y_train = torch.load(os.path.join(filePath,"y_train1.pt"))
y_test = torch.load(os.path.join(filePath,"y_test.pt"))

testset = Dataset.Dataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)

dataset = Dataset.Dataset(X_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# train_meander_loader = torch.DataLoader(train_meanders, batch_size)
# train_spiral_loader = torch.DataLoader(train_spirals, batch_size)
# train_circle_loader = torch.DataLoader(train_circles, batch_size)

device=torch.device('cuda:0')
NN = CircleConv(num_classes=1,size = (238,211)) #hardcoded for now
NN.to(device)

#TODO maybe set these as default values in constructor

optimizer = torch.optim.ASGD(params=NN.parameters(), lr=0.01) #TODO ask about lr
#torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1, last_epoch=-1) #commented out the learning rate decay // also dropped lr to 0.01
cost_func = nn.BCELoss()

for i in range(epochs):
    temp_accs = []
    temp_losses = []
    for j, (X,y) in enumerate(data_loader):
        current_batch = y.shape[0]
        X = X.to(device)
        y = y.to(device)
        yhat = NN.forward(X).reshape(current_batch) #reshaped to batchsize
        loss = cost_func(yhat, y)
        yhat = (yhat>threshold).float()
        acc = torch.eq(yhat.round(), y).float().mean()  # accuracy
        
        for pred,actual in zip(yhat.tolist(),y.tolist()):
            conf_mat[int(actual),int(pred)] += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        temp_losses.append(loss.data.item()) #was loss.data[0]
        temp_accs.append(acc.data.item()) #was acc.data[0]
        
        if j % 15 == 14:
            print("[{}/{}], loss: {} acc: {}".format(i,
                                                 epochs, np.round(loss.data.item(), 3), np.round(acc.data.item(), 3)))
    losses.append(mean(temp_losses))
    accs.append(mean(temp_accs))
    l_precision = (conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[0,1]))
    precision.append(l_precision)
    l_recall = (conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[1,0]))
    recall.append(l_recall)
    f1.append(2* ((l_precision*l_recall)/(l_precision+l_recall)))
    
    if (i+1) % 5 == 0:
        breakout = evaluate()
        if breakout >= 3:
            print("Breakout triggered at epoch",i)
            break
    NN.train()


x = list(range(len(losses)))

fig = plt.figure()
plt.plot(x,losses,color = 'r')
plt.xlabel('Minibatches')
plt.ylabel('Loss')
plt.savefig('/projectnb/riseprac/GroupB/Images/CircleLoss'+str(run_num)+'.png')

plt.plot(x,accs,color = 'g')
plt.xlabel('Minibatches')
plt.ylabel('Accuracy (dec)')
plt.savefig('/projectnb/riseprac/GroupB/Images/CircleAccuracy'+str(run_num)+'.png')

x = list(range(epochs))
plt.plot(x,precision,color='b',label = 'precision')
plt.plot(x,recall,color='r', label = 'recall')
plt.plot(x,f1,color='k',label = 'f1 score')
plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Score (%)")
plt.savefig('/projectnb/riseprac/GroupB/Images/CircleScores'+str(run_num)+'.png')


labels = ['True Neg','False Pos','False Neg','True Pos']
labels = np.asarray(labels).reshape(2,2)
x_axis_labels = ['Healthy', 'PD']
y_axis_labels = ['PD', 'Healthy']
sns_plot = sns.heatmap(conf_mat/torch.sum(conf_mat), annot=labels, fmt='.2', xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap='Blues')
plt.xlabels('Predicted Category')
plt.ylabels('True Category')
conf_img = sns_plot.get_figure()
conf_img.savefig('/projectnb/riseprac/GroupB/Images/CircleConf_matFINAL.png')

print('Avg/final loss =',mean(losses),losses[-1])
print('Avg/final accuracy =',mean(accs),accs[-1])
print('Final precision =',precision[-1])
print('Final recall =',recall[-1])
print('Final f1 =',f1[-1])


#torch.save(NN.state_dict(),'/projectnb/riseprac/GroupB/CircleState_dict'+str(run_num)+'.pt')
