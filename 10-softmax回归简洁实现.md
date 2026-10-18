nn.Flatten():默认从第一位开始展平，跳过第 0 维（batch_size）
本例把输入展平为 [样本数, 特征数] 再交给 Linear。Linear 也能接收更高维张量，它只变换最后一维。

.apply:Pytorch网络自带方法，自动遍历 net 每一层
net.apply(init_weights)：把每一层当做参数传给init_weights

本例的形状
[batch, 1, 28, 28] -> Flatten -> [batch, 784] -> Linear(784, 10) -> [batch, 10]
Flatten 不学习参数，Linear 的 weight 和 bias 才需要训练。

输出不要再接 Softmax 层
nn.CrossEntropyLoss 直接接收 logits；标签传 [batch] 的类别编号。
预测时 logits.argmax(dim=1)，dim=1 是类别那一维。

训练和评估
• net.train()：启用训练行为
• net.eval()：切换评估行为，影响 Dropout、BatchNorm 等层
• torch.no_grad()：不记录梯度，减少评估开销
• eval() 和 no_grad() 作用不同，通常评估时一起用

当前安装的 D2L 没有 train_ch3，所以对应 .py 里保留了本地训练函数。
