from architecture import SimpleConv
import getData
import Dataset
import torch
from torch import nn
import numpy as np

epochs = 15 #what value should we set this
batch_size = 5
losses = []
accs = []

X_train,X_test,y_train,y_test = getData.getData('/Users/adamwasserman/Documents/Image_Data') #creates a tuple: (data, values)

dataset = Dataset(X_train, y_train)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# train_meander_loader = torch.DataLoader(train_meanders, batch_size)
# train_spiral_loader = torch.DataLoader(train_spirals, batch_size)
# train_circle_loader = torch.DataLoader(train_circles, batch_size)

NN = SimpleConv(num_classes=1, meander_size=744*822, spiral_size=756*786, circle_size=675*720) #hardcoded for now
#TODO maybe set these as default values in constructor

optimizer = torch.optim.Adam(params=NN.parameters(), lr=0.001) #TODO ask about lr
cost_func = nn.BCELoss()

for i in range (epochs):
    for j, data in enumerate(dataset):
        yhat = NN.forward(data[0][:, 0], data[0][:, 1], data[0][:, 2])
        loss = cost_func(yhat, data[1])
        yhat = (yhat>0.5).float()
        acc = torch.eq(yhat.round(), data[1]).float().mean()  # accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.data[0])
        accs.append(acc.data[0])

        if j % 5 == 1:
            print("[{}/{}], loss: {} acc: {}".format(i,
                                                 epochs, np.round(loss.data[0], 3), np.round(acc.data[0], 3)))