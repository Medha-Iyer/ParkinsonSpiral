from architecture import SimpleConv
import Dataset
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

epochs = 100 #what value should we set this
batch_size = 5
threshold = 0.5
run_num = 1
losses = []
accs = []
precision = []
recall = []
f1 = []

conf_mat = torch.zeros(2,2)

filePath = '/projectnb/riseprac/GroupB'

X_train = torch.load(os.path.join(filePath,"X_train.pt"))
X_test = torch.load(os.path.join(filePath,"X_test.pt"))
y_train = torch.load(os.path.join(filePath,"y_train.pt"))
y_test = torch.load(os.path.join(filePath,"y_test.pt"))

dataset = Dataset.Dataset(X_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# train_meander_loader = torch.DataLoader(train_meanders, batch_size)
# train_spiral_loader = torch.DataLoader(train_spirals, batch_size)
# train_circle_loader = torch.DataLoader(train_circles, batch_size)

device1=torch.device('cuda:0')
device2=torch.device('cuda:1')


NN = SimpleConv(num_classes=1,size = (756,822)) #hardcoded for now
#NN.to(device)

#TODO maybe set these as default values in constructor

optimizer = torch.optim.Adam(params=NN.parameters(), lr=0.05) #TODO ask about lr
torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1, last_epoch=-1)
cost_func = nn.BCELoss()

for i in range (epochs):
    for j, (X,y) in enumerate(data_loader):
        #X = X.to(device)
        y = y.to(device2)
        yhat = NN.forward(X[:, 0].to(device1), X[:, 1].to(device2), X[:, 2].to(device1)).reshape(batch_size) #reshaped to batchsize
        loss = cost_func(yhat, y)
        yhat = (yhat>threshold).float()
        acc = torch.eq(yhat.round(), y).float().mean()  # accuracy
        
        for pred,actual in zip(yhat.tolist(),y.tolist()):
            conf_mat[int(actual),int(pred)] += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data.item()) #was loss.data[0]
        accs.append(acc.data.item()) #was acc.data[0]
        if (j+1)%10 == 0:
            print("[{}/{}], loss: {} acc: {}".format(i,
                                                 epochs, np.round(loss.data[0], 3), np.round(acc.data[0], 3)))
    precision.append((conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[0,1])))
    recall.append((conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[1,0])))
    f1.append(2* ((precision*recall)/(precision+recall)))


x = list(range(len(losses)))

fig = plt.figure()
plt.plot(x,losses,color = 'r')
plt.x_label('Minibatches')
plt.y_label('Loss')
plt.savefig('./images/loss'+run_num+'.png')

plt.plot(x,acc,color = 'g')
plt.y_label('Accuracy (dec)')
plt.savefig('./images/accuracy'+run_num+'.png')

x = list(range(epochs))
plt.plot(x,precision,color='b',label = 'precision')
plt.plot(x,recall,color='r', label = 'recall')
plt.plot(x,f1,color='k',label = 'f1 score')

plt.x_label("Epoch")
plt.y_label("Score (%)")
plt.savefig('./images/scores'+run_num+'.png')

torch.save('./images/scores.npy', conf_mat)

torch.save(NN.state_dict(),'/projectnb/riseprac/GroupB/state_dict'+str(run_num)+'.pt')

    
