from architecture import SimpleConv
import getData
import Dataset
import torch
from torch import nn

epochs = 15 #what value should we set this
batch_size = 5
X_train,X_test,y_train,y_test = getData.getData('/Users/adamwasserman/Documents/Image_Data') #creates a tuple: (data, values)

train_meanders = Dataset(X_train[:,0], y_train)
train_spirals = Dataset(X_train[:,1], y_train)
train_circles = Dataset(X_train[:,2], y_train)

dataset = torch.Dataset(X_train, y_train)


# train_meander_loader = torch.DataLoader(train_meanders, batch_size)
# train_spiral_loader = torch.DataLoader(train_spirals, batch_size)
# train_circle_loader = torch.DataLoader(train_circles, batch_size)

NN = SimpleConv(num_classes=1, meander_size=744*822, spiral_size=756*786, circle_size=675*720) #hardcoded for now
#TODO maybe set these as default values in constructor

optimizer = torch.optim.Adam(params=NN.parameters(), lr=0.001) #TODO ask about lr
cost_func = nn.CrossEntropyLoss()

for i in range (epochs):
    for j, data in enumerate(zip(train_meanders, train_spirals, train_circles)):
        NN.forward(data[0], data[1], data[2])