import torch
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    # 对输入的每个通道分别做二维互相关，再将结果相加
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    # 对每个输出通道分别计算
    return torch.stack([corr2d_multi_in(X, k) for k in K], dim=0)


# 多输入、多输出通道示例
X = torch.normal(0, 1, (2, 3, 3))
K = torch.normal(0, 1, (2, 2, 2))
print(corr2d_multi_in(X, K))

K_multi_out = torch.stack((K, K + 1, K + 2), dim=0)
print(corr2d_multi_in_out(X, K_multi_out))


# 1 x 1 卷积
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o, k_c_i, k_h, k_w = K.shape

    if (k_c_i, k_h, k_w) != (c_i, 1, 1):
        raise ValueError("K 的形状必须是 (输出通道数, 输入通道数, 1, 1)")

    X = X.reshape(c_i, h * w)
    K = K.reshape(c_o, c_i)
    Y = torch.matmul(K, X)
    return Y.reshape(c_o, h, w)


X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert torch.allclose(Y1, Y2, atol=1e-6)