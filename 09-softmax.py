import matplotlib
matplotlib.use('MacOSX')

import torch
from IPython import display
from d2l import torch as d2l

#展平图像，视为长度 784（1*28*28） 的向量。数据集有十个类别所以输出的维度是 10
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

#实现 softmax 回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


#计算正确的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    #新建实例的时候传入几个n，代表要同时维护几个累加的数
    def __init__(self, n):
        self.data = [0.0] * n

    #传入数字，把旧值和新值一一相加
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    #把数字归零
    def reset(self):
        self.data = [0.0] * len(self.data)

    #可以让metric.data[0]简写为metric[0]
    def __getitem__(self, idx):
        return self.data[idx]
    
#评估任意模型 net 的准确率
def evaluate_accuracy(net, data_iter):
    #计算在指定数据集上模型的精度
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式,关闭训练行为,不要更新模型
    metric = Accumulator(2)  # 正确预测数、预测总数
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Animator:
    #在动画中绘制数据
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        d2l.plt.draw()
        d2l.plt.pause(0.01)

#只跑一个 epoch
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.sum().backward()
            updater.step()
            metric.add(float(l.sum()), accuracy(y_hat, y),
                       y.size().numel())
        else:
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum().detach()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]

#反复调用train_epoch_ch3
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    # Animator:D2L自己写的画图工具,动态绘图,可视化曲线
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    #循环：跑 num_epochs 个训练轮次
    for epoch in range(num_epochs):
        #调用【单轮训练函数】train_epoch_ch3,跑完1整个训练集
        # 返回 (train_loss, train_acc) 本轮训练损失、训练准确率
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)

        #拿当前训练好的模型，在【测试集】算测试集准确率（看不见的新数据）
        test_acc = evaluate_accuracy(net, test_iter)

        #把这一轮的3个指标加到图上,动态画曲线
        animator.add(epoch + 1, train_metrics + (test_acc,))

lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

    
def predict_ch3(net, test_iter, n=6):
    """预测标签（定义见第3章）。"""
    for X, y in test_iter:
        break   # 只取出第一个batch,直接跳出循环,不用遍历全部测试集
    
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28,28)), 1, n, titles=titles[0:n])


# macOS 多进程加载数据时，子进程不能再次执行训练入口。
if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()
