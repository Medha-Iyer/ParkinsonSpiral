from architecture import SimpleConv
import CombDataset
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statistics import mean

epochs = 1500  # what value should we set this
batch_size = 10
threshold = 0.5
run_num = 1
temp_accs = []
temp_losses = []
losses = []
accs = []
precision = []
recall = []
f1 = []

conf_mat = torch.zeros(2, 2)

filePath = '/projectnb/riseprac/GroupB/preprocessedData'

Xm = torch.load(os.path.join(filePath, "Xm_train.pt"))
Xs = torch.load(os.path.join(filePath, "Xs_train.pt"))
Xc = torch.load(os.path.join(filePath, "Xc_train.pt"))
y_train = torch.load(os.path.join(filePath, "y_train.pt"))


dataset = CombDataset.Dataset(Xm,Xs,Xc,y_train)
data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

# train_meander_loader = torch.DataLoader(train_meanders, batch_size)
# train_spiral_loader = torch.DataLoader(train_spirals, batch_size)
# train_circle_loader = torch.DataLoader(train_circles, batch_size)


device=torch.device('cuda:0')

NN = SimpleConv(num_classes=1,size = (756,822)) #hardcoded for now
NN.to(device)


optimizer = torch.optim.ASGD(params=NN.parameters())  # TODO ask about lr
cost_func = nn.BCELoss()


for i in range (epochs):
    for j, (Xme,Xsp,Xci,y) in enumerate(data_loader):
        current_batch = y.shape[0]
        Xme,Xsp,Xci = Xme.to(device), Xsp.to(device), Xci.to(device)
        y = y.to(device)
        yhat = NN.forward(Xme, Xsp, Xci).reshape(current_batch) #reshaped to batchsize
        loss = cost_func(yhat, y)
        yhat = (yhat > threshold).float()
        acc = torch.eq(yhat.round(), y).float().mean()  # accuracy

        for pred, actual in zip(yhat.tolist(), y.tolist()):
            conf_mat[int(actual), int(pred)] += 1
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


x = list(range(len(losses)))

fig = plt.figure()

plt.plot(x,losses,color = 'r')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('/projectnb/riseprac/GroupB/Images/MAINloss'+str(run_num)+'.png')

plt.plot(x,accs,color = 'g')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (dec)')
plt.savefig('/projectnb/riseprac/GroupB/Images/MAINaccuracy'+str(run_num)+'.png')

x = list(range(epochs))
plt.plot(x,precision,color='b',label = 'precision')
plt.plot(x,recall,color='r', label = 'recall')
plt.plot(x,f1,color='k',label = 'f1 score')
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Score (%)")
plt.savefig('/projectnb/riseprac/GroupB/Images/MAINscores'+str(run_num)+'.png')

sns_plot = sns.heatmap(conf_mat/torch.sum(conf_mat), annot=True,
            fmt='.2%', cmap='Blues')
conf_img = sns_plot.get_figure()    
conf_img.savefig('/projectnb/riseprac/GroupB/Images/MAINconf_mat' + str(run_num)+ '.png')

torch.save(NN.state_dict(), '/projectnb/riseprac/GroupB/MAINstate_dict' + str(run_num) + '.pt')
