import torch
from torch import nn
from torch.nn import functional as F

#加载和保存张量
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load("x-file")

# 存储张量列表
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')

# 存储张量字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')

# 加载保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

torch.save(net.state_dict(), 'mlp.params')

clone = MLP() # 新建一个和原来net结构一模一样的MLP网络
clone.load_state_dict(torch.load("mlp.params")) # 从文件读取参数，加载进这个新网络
clone.eval() # 设为评估模式,关闭训练

Y_clone = clone(X)
print(Y_clone == Y)