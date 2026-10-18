import torch
from torch import nn
from d2l import torch as d2l

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 推理模式：用训练好的全局滑动均值/方差
    if not torch.is_grad_enabled():
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 仅支持2维(全连接输出) 和 4维(卷积输出)
        assert len(X.shape) in (2, 4)
        
        # 情况1:全连接层的BN, shape=(batch, 特征数)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
            
        # 情况2:卷积层的BN,shape=(batch, 通道, 高, 宽)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)

        # 用当前batch的均值方差做归一化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 滑动更新全局均值方差，留作推理用
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    # 可学习参数:缩放gamma + 平移beta
    Y = gamma * X_hat + beta
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        super().__init__()
        # 根据维度确定参数形状
        if num_dims == 2:
            shape = (1, num_features)       # 全连接层BN:(batch, 特征数)
        else:
            shape = (1, num_features, 1, 1) # 卷积层BN:(batch, 通道, 高, 宽)

        # 可学习参数:缩放gamma、平移beta
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # 全局滑动统计量：非可学习参数，训练时更新、推理时固定
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 设备对齐：确保统计量和输入在同一个设备上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        # 调用批量归一化核心计算
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9
        )
        return Y

# 在 LeNet 的卷积层、全连接层之后加入批量归一化，再使用激活函数
# 输入为 Fashion-MNIST 原始的 28×28 灰度图
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5),
    BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 120),
    BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84),
    BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10)
)


if __name__ == '__main__':
    # 使用主进程加载数据，避免 Mac 子进程重复执行脚本
    d2l.get_dataloader_workers = lambda: 0

    # 逐层检查输出形状，不记录梯度、不更新滑动统计量
    with torch.no_grad():
        X = torch.rand(size=(1, 1, 28, 28))
        for layer in net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        raise RuntimeError('当前环境没有可用 GPU')

    print('训练设备：', device, flush=True)

    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)

    d2l.plt.show()
