import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)

#提取想要的层的参数
print(net[2].state_dict())

print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)

print(net[2].weight.grad == None) #True 还没有开始反向传播

#一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

#直接通过名字访问
net.state_dict()['2.bias'].data

#从嵌套块收集参数
import torch
from torch import nn

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4),
                         nn.ReLU())
#嵌套四个 block1
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# rgnet(X)
print(rgnet) #大致了解网络长什么样子

#参数初始化
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01) # 函数末尾名字下划线：原地 inplace操作
        nn.init.zeros_(m.bias)

net.apply(init_normal) # 遍历网络所有子模块
print(net[0].weight.data[0], net[0].bias.data[0])

#不能把参数全部初始化为常数：同一层所有神经元的几串结果完全一样
#要是只有 bias 不一样依旧不行：向前计算后只是相互平移的版本，没有独立的特征方向，反向传播梯度依然相等

#对某些块应用不同的初始化方法
def xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

#可以设置对某一块单独用某个初始化
net[0].apply(xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)

#自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print(
            "Init",
            *[(name, param.shape) for name, param in m.named_parameters()][0]  #named_parameters():nn.Moddule自带方法，返回所有可训练参数
        )
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(net[0].weight[:2])

#参数绑定
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                    nn.ReLU(), nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])