import torch
from torch import nn


def pool2d(X, pool_size, mode="max"):
    p_h, p_w = pool_size
    out_h = X.shape[0] - p_h + 1
    out_w = X.shape[1] - p_w + 1
    Y = torch.zeros((out_h, out_w), dtype=X.dtype, device=X.device)

    for i in range(out_h):
        for j in range(out_w):
            window = X[i:i + p_h, j:j + p_w]
            if mode == "max":
                Y[i, j] = window.max()
            elif mode == "avg":
                Y[i, j] = window.mean()
            else:
                raise ValueError("mode 必须是 'max' 或 'avg'")

    return Y


# 自定义池化函数示例
X_2d = torch.arange(16, dtype=torch.float32).reshape(4, 4)
print(pool2d(X_2d, (3, 3), mode="max"))
print(pool2d(X_2d, (3, 3), mode="avg"))

# 使用 PyTorch 池化层
X = X_2d.reshape(1, 1, 4, 4)
max_pool = nn.MaxPool2d(3)
max_pool_with_padding = nn.MaxPool2d(3, padding=1, stride=2)

# 池化层在每个通道上单独运算
X = torch.cat((X, X + 1), dim=1)
print(max_pool(X))
print(max_pool_with_padding(X))