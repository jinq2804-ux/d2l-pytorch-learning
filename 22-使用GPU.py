import torch
from torch import nn


# Apple 芯片通过 MPS 使用 GPU；NVIDIA 显卡通过 CUDA 使用 GPU。
def try_gpu(i=0):
    """返回第 i 个可用 GPU；没有对应 GPU 时返回 CPU。"""
    if i < 0:
        raise ValueError("GPU 编号不能为负数")
    if torch.cuda.is_available() and torch.cuda.device_count() > i:
        return torch.device(f"cuda:{i}")
    # M2 的 MPS 后端提供一个 GPU 设备，没有 mps:1。
    if i == 0 and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def try_all_gpus():
    """返回所有可用 GPU；没有 GPU 时返回 [cpu]。"""
    if torch.cuda.is_available():
        return [torch.device(f"cuda:{i}")
                for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        return [torch.device("mps")]
    return [torch.device("cpu")]


if __name__ == "__main__":
    # 1. 查询设备：Mac 不需要运行 nvidia-smi，也不需要安装 CUDA。
    device = try_gpu()
    devices = try_all_gpus()
    print("CUDA 可用:", torch.cuda.is_available())
    print("MPS 可用:", torch.backends.mps.is_available())
    print("GPU 数量:", sum(d.type != "cpu" for d in devices))
    print("当前设备:", device)
    print("第 10 号 GPU（不存在时返回 CPU）:", try_gpu(10))
    print("所有可用设备:", devices)

    # 2. 直接在 GPU 上创建张量；没有可用 GPU 时会使用 CPU。
    X = torch.ones(2, 3, device=device)
    print("X:", X)
    print("X 设备:", X.device)

    # 3. M2 只有一个 MPS 设备，用 CPU → GPU 演示设备迁移。
    Y = torch.rand(2, 3, device="cpu")
    print("Y 原来的设备:", Y.device)
    Z = Y.to(device)
    print("Z 迁移后的设备:", Z.device)

    # 张量必须位于同一设备才能相加。
    result = X + Z
    print("X + Z 结果:", result)
    print("结果设备:", result.device)

    # 张量已在目标设备上时，.to(device) 返回自身。
    print("Z.to(device) is Z:", Z.to(device) is Z)

    # 4. 将网络迁移到与输入相同的设备，执行前向传播。
    net = nn.Sequential(nn.Linear(3, 1)).to(device)
    output = net(X)
    print("网络输出:", output)
    print("模型参数设备:", net[0].weight.device)

    # 如需转成 NumPy，应先移回 CPU；detach() 去掉梯度跟踪。
    print("输出转为 NumPy:", output.detach().cpu().numpy())
