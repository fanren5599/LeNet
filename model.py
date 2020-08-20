import torch.nn as nn
import torch

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.base=nn.Sequential(nn.Conv2d(1,6,kernel_size=3),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2),nn.Conv2d(6,16,kernel_size=3),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2))
        self.lin1=nn.Linear(16*6*6,120)
        self.lin2=nn.Linear(120,84)
        self.lin3=nn.Linear(84,10)
    def forward(self, x):
        x=self.base(x)
        x=x.view(x.size(0),-1)
        x=torch.relu(self.lin1(x))
        x=torch.relu(self.lin2(x))
        x=self.lin3(x)
        return x
if __name__=="__main__":
    model=LeNet()
    ret=model(torch.randn(1,1,32,32))
    print(ret.shape)
