# 导入依赖库
import sys
import matplotlib
import torch
from torch import nn
from d2l import torch as d2l


# 从零实现 Dropout：每个元素以 dropout 的概率置零。
# 保留的元素除以 1-dropout，使输出的期望保持不变。
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X)
    if dropout == 0:
        return X
    mask = (torch.rand_like(X) > dropout).to(X.dtype)
    return mask * X / (1.0 - dropout)


# 超参数定义
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.8, 0.8
num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss()


# 自定义带 Dropout 的两层 MLP；train()/eval() 控制 self.training。
class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        if self.training:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)


# 当前安装的 d2l 没有 train_ch3，使用本地训练与评估函数。
# 每轮训练前 net.train()，评估前 net.eval()，确保测试时关闭 Dropout。
def train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer, title='Dropout MLP'):
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
        ax.set(xlabel='epoch', xlim=(1, max(2, num_epochs)), title=title)
        ax.legend()
        fig.tight_layout()
        plt.pause(0.01)
    return history


# macOS 子进程重新导入脚本时，不重复加载数据和训练。
if __name__ == "__main__":
    if sys.platform == "darwin":
        matplotlib.use("MacOSX")
    import matplotlib.pyplot as plt

    # 测试 Dropout 层效果；0.5 时具体哪些位置置零是随机的。
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print(X)
    print(dropout_layer(X, 0.0))
    print(dropout_layer(X, 0.5))
    print(dropout_layer(X, 1.0))

    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    print('开始训练：从零实现 Dropout', flush=True)
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer,
              title='Custom Dropout')

    print('开始训练：PyTorch 内置 Dropout', flush=True)
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(256, 10),
    )
    net.apply(init_weights)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer,
              title='PyTorch Dropout')
    plt.show()  # 两个实验结束后保持图形窗口打开。

# 若做无 Dropout 对照：将 dropout1、dropout2 改为 0.0，再重新运行。
