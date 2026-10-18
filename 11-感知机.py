import torch
from torch import nn
from d2l import torch as d2l

num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

params = [W1, b1, W2, b2]

#实现 ReLU 激活函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

#实现模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

loss = nn.CrossEntropyLoss()

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


# macOS 的子进程重新导入脚本时，不重复启动数据加载和训练。
if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
