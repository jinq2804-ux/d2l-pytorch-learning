import sys
import matplotlib

# 在 macOS 终端直接运行时使用原生窗口；导入脚本时不强制切换后端。
if __name__ == "__main__" and sys.platform == "darwin":
    matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l

#生成人造数据集
#训练样本只有20个，特征却有200个，制造过拟合场景
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

#初始化参数w,b
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

#定义L2惩罚项
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

#训练函数，传入lambda控制正则强度
def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    # 使用窗口绘图，避免依赖 Jupyter 的 display/clear_output。
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set(xlabel='epochs', ylabel='loss', yscale='log',
           xlim=(5, num_epochs), title=f'L2 regularization: lambda={lambd}')
    train_line, = ax.plot([], [], label='train')
    test_line, = ax.plot([], [], label='test')
    ax.legend()
    epochs, train_losses, test_losses = [], [], []
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with torch.enable_grad():
                # 总损失 = 原始MSE损失 + λ * L2惩罚项
                l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            # 评估不需要梯度；evaluate_loss 的第三个参数是损失函数。
            with torch.no_grad():
                train_loss = d2l.evaluate_loss(net, train_iter, loss)
                test_loss = d2l.evaluate_loss(net, test_iter, loss)
            epochs.append(epoch + 1)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_line.set_data(epochs, train_losses)
            test_line.set_data(epochs, test_losses)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            fig.tight_layout()
            plt.pause(0.01)  # 处理窗口事件并刷新曲线
            print(f'epoch {epoch + 1}: train loss={train_loss:.6f}, '
                  f'test loss={test_loss:.6f}', flush=True)
    print('w的L2范数是：', torch.norm(w).item())

if __name__ == "__main__":
    train(lambd=10)
    plt.show()  # 训练完成后保持窗口打开，关闭窗口后程序才退出。
