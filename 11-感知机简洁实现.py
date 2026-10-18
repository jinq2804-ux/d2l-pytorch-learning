import torch
from torch import nn
from d2l import torch as d2l

# 多层感知机：输入 784 个特征，隐藏层 256 个单元，输出 10 个类别。
net = nn.Sequential(
    nn.Flatten(),        # [batch, 1, 28, 28] -> [batch, 784]
    nn.Linear(784, 256), # H = X @ weight.T + bias
    nn.ReLU(),          # H = max(H, 0)，引入非线性
    nn.Linear(256, 10),  # 输出 logits：[batch, 10]
)


def init_weights(m):
    """与从零实现保持相同的初始化分布：小随机权重、零偏置。"""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        nn.init.zeros_(m.bias)


# apply 会遍历子模块；只初始化其中的 Linear 层。
net.apply(init_weights)

# 直接传入 logits；CrossEntropyLoss 内部完成 log-softmax 和交叉熵计算。
# 默认返回每个 batch 的平均损失。
loss = nn.CrossEntropyLoss()


# 当前安装的 d2l 没有 train_ch3，因此在本文件提供训练函数。
def train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer):
    """训练分类模型（支持普通函数和 nn.Module）；loss 使用默认的 reduction='mean'。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    history = []
    for epoch in range(num_epochs):
        if isinstance(net, nn.Module):
            net.train()
        loss_sum, correct, total = 0.0, 0, 0
        for X, y in train_iter:
            trainer.zero_grad()
            logits = net(X)
            batch_loss = loss(logits, y)
            batch_loss.backward()
            trainer.step()
            loss_sum += batch_loss.item() * y.numel()
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.numel()

        if isinstance(net, nn.Module):
            net.eval()
        test_correct, test_total = 0, 0
        with torch.no_grad():
            for X, y in test_iter:
                test_correct += (net(X).argmax(dim=1) == y).sum().item()
                test_total += y.numel()

        train_loss = loss_sum / total
        train_acc = correct / total
        test_acc = test_correct / test_total
        history.append((train_loss, train_acc, test_acc))
        print(f'epoch {epoch + 1}/{num_epochs}: loss={train_loss:.4f}, '
              f'train acc={train_acc:.4f}, test acc={test_acc:.4f}', flush=True)

        ax.clear()
        epochs = range(1, epoch + 2)
        for i, label in enumerate(('train loss', 'train acc', 'test acc')):
            ax.plot(epochs, [values[i] for values in history], label=label)
        ax.set(xlabel='epoch', xlim=(1, max(2, num_epochs)))
        ax.legend()
        fig.tight_layout()
        plt.pause(0.01)
    plt.show()
    return history


# macOS DataLoader 使用多进程时，子进程不能重复执行训练入口。
if __name__ == "__main__":
    batch_size, lr, num_epochs = 256, 0.1, 10
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # net.parameters() 自动收集两层 Linear 的权重和偏置。
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
