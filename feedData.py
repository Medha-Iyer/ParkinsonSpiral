from architecture import SimpleConv
import getData
import Dataset
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

epochs = 100 #what value should we set this
batch_size = 5
threshold = 0.5
losses = []
accs = []
precision = []
recall = []
f1 = []

conf_mat = np.zeros((2,2), dtype = 'i')

X_train,X_test,y_train,y_test = getData.getData('/Users/adamwasserman/Documents/RISE/Project')

dataset = Dataset.Dataset(X_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# train_meander_loader = torch.DataLoader(train_meanders, batch_size)
# train_spiral_loader = torch.DataLoader(train_spirals, batch_size)
# train_circle_loader = torch.DataLoader(train_circles, batch_size)

NN = SimpleConv(num_classes=1,size = (756,822)) #hardcoded for now
#TODO maybe set these as default values in constructor

optimizer = torch.optim.Adam(params=NN.parameters(), lr=0.05) #TODO ask about lr
torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1, last_epoch=-1, verbose=False)
cost_func = nn.BCELoss()

for i in range (epochs):
    for j, data in enumerate(data_loader):
        yhat = NN.forward(data[0][:, 0], data[0][:, 1], data[0][:, 2]).reshape(5) #reshaped to 5
        loss = cost_func(yhat, data[1])
        yhat = (yhat>threshold).float()
        acc = torch.eq(yhat.round(), data[1]).float().mean()  # accuracy
        
        for pred,actual in zip(yhat,data[1]):
            conf_mat[actual,pred] += 1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data.item()) #was loss.data[0]
        accs.append(acc.data.item()) #was acc.data[0]
        if j % 5 == 1:
            print("[{}/{}], loss: {} acc: {}".format(i,
                                                 epochs, np.round(loss.data[0], 3), np.round(acc.data[0], 3)))
    precision.append((conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[0,1])))
    recall.append((conf_mat[1,1])/((conf_mat[1,1]) + (conf_mat[1,0])))
    f1.append(2* ((precision*recall)/(precision+recall)))


x = range(len(losses))

fig = plt.figure()
plt.plot(x,losses,color = 'r')
plt.x_label('Minibatches')
plt.y_label('Loss')
plt.show()

plt.plot(x,acc,color = 'g')
plt.y_label('Accuracy (dec)')
plt.show()

x = range(epochs)
plt.plot(x,precision,color='b',label = 'precision')
plt.plot(x,recall,color='r', label = 'recall')
plt.plot(x,f1,color='k',label = 'f1 score')

plt.x_label("Epoch")
plt.y_label("Score (%)")
plt.show()


print('Confusion matrix:', '\n',conf_mat[0],'\n',conf_mat[1])

torch.save(NN.state_dict(),'/projectnb/riseprac/GroupB/state_dict.pt')

    