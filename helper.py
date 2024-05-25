import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import torch

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #classifier
        self.linear_1 = torch.nn.Linear(32*8*4*8,32*8*4)
        self.linear_2 = torch.nn.Linear(32*8*4,32*8)
        self.batch_norm_1 = torch.nn.BatchNorm1d(32*8)

        self.linear_3 = torch.nn.Linear(32*8,32)
        self.linear_4 = torch.nn.Linear(32,1)

        self.activation = torch.nn.ReLU()


    def forward(self,x):
        x = x.reshape(x.shape[0], -1)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.activation(x)
        x = self.batch_norm_1(x)

        x = self.linear_3(x)
        x = self.activation(x)
        x = self.linear_4(x)
        return x
    
net = Classifier().cuda()
inp = torch.rand(2,32,8,4,8).cuda()
gt = torch.rand(2,1).cuda()
outp = net(inp)
loss = torch.nn.BCEWithLogitsLoss()(outp, gt)
loss.backward()
