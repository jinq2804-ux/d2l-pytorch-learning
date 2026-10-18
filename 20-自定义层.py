import torch
import torch.nn.functional as F
from torch import nn

#构造一个没有任何参数的自定义层
#自定义向前传播方法
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))

#构造一个带参数的自定义层
import torch
from torch import nn
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)

dense = MyLinear(5, 3)
dense.weight

#nn.Parameter：把普通张量包装成 可训练参数
#放在继承nn.Module的类里面，父类nn.Module会自动识别、注册这个变量，把它当成网络参数，自动开启梯度、放进parameters()、named_parameters()里，训练的时候可以更新。