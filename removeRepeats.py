import torch
Xm_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xm_test.pt')
Xs_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xs_test.pt')
Xc_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/Xc_test.pt')
y_test = torch.load('/projectnb/riseprac/GroupB/preprocessedData/y_test.pt')

Xm_test,Xs_test,Xc_test,y_test = Xm_test[4:48],Xs_test[4:48],Xc_test[4:48],y_test[4:48]

torch.save(Xm_test,'/projectnb/riseprac/GroupB/preprocessedData/Xm_test1.pt')
torch.save(Xs_test,'/projectnb/riseprac/GroupB/preprocessedData/Xs_test1.pt')
torch.save(Xc_test,'/projectnb/riseprac/GroupB/preprocessedData/Xc_test1.pt')
torch.save(y_test,'/projectnb/riseprac/GroupB/preprocessedData/y_test1.pt')