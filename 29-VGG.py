import torch
from torch import nn
from d2l import torch as d2l

# 直接运行此脚本时使用主进程，避免 Mac 子进程重复启动训练
d2l.get_dataloader_workers = lambda: 0

#创建卷积层
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for _ in range(num_convs):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

#创建VGG网络
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)

X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__, 'output shape:\t', X.shape)

ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    raise RuntimeError('当前环境没有可用 GPU')

print('训练设备：', device)

lr, num_epochs, batch_size = 0.05, 10, 32
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)

# 在普通 Python 脚本中显示图形窗口
d2l.plt.show()
