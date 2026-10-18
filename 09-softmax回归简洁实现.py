import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

#初始化权重
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

#在交叉熵函数中传递未归一化的预测，并同时计算 softmax 及其对数
loss = nn.CrossEntropyLoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer):
    """训练 softmax 分类器；loss 使用默认的 reduction='mean'。"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    history = []
    for epoch in range(num_epochs):
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


# macOS 的数据加载子进程重新导入脚本时，不重复启动训练。
if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

