import torch
from torch import nn
from torch.nn import functional as F

#通过继承nn.module类来进行模型构造
#init里定义有哪些层，forward里定义如何进行前向运算

#最简易写法
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)

net(X)

#自定义 MLP 类
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

#自己手写简易版sequential
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for i, block in enumerate(args):
            self._modules[str(i)] = block
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    
#nn.Sequential只能按放入顺序执行前向传播，因此可以自定义代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
# net(X)