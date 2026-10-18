import torch
import numpy as np

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())

y = x.reshape(3,4)
print(y)

m = torch.zeros((2,3,4))
print(m)

n=torch.ones((2,3,4))
print(n)

X = torch.arange(12,dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
M = torch.cat((X,Y),dim=0)
N = torch.cat((X,Y),dim=1)
print(M)
print(N)

print(X[-1])

X[0:2,:] = 12
print(X)

#新结果内存分配 
before = id(Y)
Y = Y + X
print(id(Y) == before) #False

#原地操作 
before = id(X)
X += Y
print(id(X) == before) #False

Z = torch.zeros_like(Y)
before = id(Z)
Z[:] = X + Y
print(id(Z) == before) #True

#转换为 Numpy 张量
A = X.numpy()
B = torch.tensor(A)
print(type(A),type(B)) #<class 'numpy.ndarray'> <class 'torch.Tensor'>

#大小为 1 的张量可以转化为 Python 张量
a = torch.tensor([3.5])
print(a,a.item(),float(a),int(a)) #tensor([3.5000]) 3.5 3.5 3

