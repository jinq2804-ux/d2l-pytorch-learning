import torch

x = torch.arange(4.0)

x.requires_grad_(True) #等价 x = torch.arange(4.0, requires_grad=True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y) #tensor(28., grad_fn=<MulBackward0>) MulBackward0保存了最后一步的求导公式，用于反向传播

y.backward()
print(x.grad)

#Pytorch默认累计梯度，我们需要清除之前的值
x.grad.zero_()

#对非标量调用 backward 需要传入 gradient 参数，该参数指定微分函数
x.grad.zero_()
y = x * x 

#将某些计算移动到记录的计算图之外
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u) #tensor([True, True, True, True])

x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x) #tensor([True, True, True, True])

#通过 Python 控制流构建的函数依然可以通过计算得到梯度(饮食构造好处的体现)
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad == d / a) #tensor(True)

