"""Fashion-MNIST LeNet：在 Apple M2 GPU 上训练、验证、保存模型并绘图。

直接运行：训练 40 轮，显示实时曲线，结束后保留窗口。
复用已训练模型：python 27-LeNet.py --evaluate --no-show
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class LeNet(nn.Module):
    """保留两层卷积、两次池化、三层全连接的 LeNet 骨架。"""
    def __init__(self, original=False):
        super().__init__()
        self.original = original
        if original:
            self.net = nn.Sequential(
                nn.Conv2d(1, 6, 5, padding=2), nn.Sigmoid(), nn.AvgPool2d(2),
                nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.AvgPool2d(2),
                nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
        else:
            # 这是改进版 LeNet，不是课本原版的 6/16 通道和 Sigmoid。
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(), nn.Linear(64 * 5 * 5, 120), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(120, 84), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(84, 10))

    def forward(self, x):
        return self.net(x)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders(args):
    # 保持课本的 ToTensor() 预处理，不加入尚未学习的数据增强。
    train_transform = transforms.ToTensor()
    eval_transform = transforms.ToTensor()
    train_data = datasets.FashionMNIST(args.data_dir, train=True,
                                       transform=train_transform, download=True)
    val_data = datasets.FashionMNIST(args.data_dir, train=True,
                                     transform=eval_transform, download=True)
    test_data = datasets.FashionMNIST(args.data_dir, train=False,
                                      transform=eval_transform, download=True)
    # 固定 54,000 / 6,000 划分，验证集用于选择模型，测试集仅用于最终评估。
    indices = torch.randperm(len(train_data),
                             generator=torch.Generator().manual_seed(args.seed))
    options = dict(batch_size=args.batch_size, num_workers=args.workers,
                   persistent_workers=args.workers > 0)
    train_loader = DataLoader(Subset(train_data, indices[6000:].tolist()),
                              shuffle=True, **options)
    val_loader = DataLoader(Subset(val_data, indices[:6000].tolist()), **options)
    test_loader = DataLoader(test_data, **options)
    return train_loader, val_loader, test_loader


@torch.inference_mode()
def evaluate(net, loader, device):
    net.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        loss_sum += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return loss_sum / total, correct / total


def draw_history(fig, axes, history):
    epochs = [row["epoch"] for row in history]
    for ax in axes:
        ax.clear()
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0].plot(epochs, [r["val_loss"] for r in history], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, [r["train_acc"] for r in history], label="train (dropout enabled)")
    axes[1].plot(epochs, [r["val_acc"] for r in history], label="validation")
    axes[1].axhline(0.9, color="gray", linestyle=":", label="90% target")
    axes[1].set(ylabel="accuracy", ylim=(0.5, 1.0), title="Accuracy")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--original", action="store_true", help="严格使用原版 LeNet 结构")
    parser.add_argument("--no-show", action="store_true", help="保存曲线但不弹窗")
    parser.add_argument("--evaluate", action="store_true", help="加载最佳模型，仅评估测试集")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "lenet_results")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs 必须至少为 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = choose_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_lenet.pt"
    print(f"device={device}; output={args.output_dir}", flush=True)
    train_loader, val_loader, test_loader = make_loaders(args)
    net = LeNet(original=args.original).to(device)
    history = []
    if not args.evaluate:
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if not args.no_show:
            plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fig.suptitle("LeNet / Fashion-MNIST")
        fig.canvas.manager.set_window_title("LeNet training (M2 GPU)")
        if not args.no_show:
            plt.show(block=False)
        # Xavier 初始化、SGD、权重衰退都是已学习过的内容。
        def init_weights(layer):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        net.apply(init_weights)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                     weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss()
        best_val_acc, best_epoch = -1.0, 0
        start = time.perf_counter()
        for epoch in range(1, args.epochs + 1):
            # 前半程大步学习，后半程把学习率调小，细调参数。
            current_lr = args.lr
            if epoch > args.epochs * 0.75:
                current_lr *= 0.1
            elif epoch > args.epochs * 0.5:
                current_lr *= 0.3
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            net.train()
            loss_sum, correct, total = 0.0, 0, 0
            epoch_start = time.perf_counter()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = net(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * labels.numel()
                correct += (logits.detach().argmax(1) == labels).sum().item()
                total += labels.numel()
            val_loss, val_acc = evaluate(net, val_loader, device)
            row = dict(epoch=epoch, train_loss=loss_sum / total,
                       train_acc=correct / total, val_loss=val_loss,
                       val_acc=val_acc, lr=optimizer.param_groups[0]["lr"])
            history.append(row)
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                # 保存 CPU 权重，可在别的设备加载；只按验证集选择模型。
                torch.save(dict(model_state={k: v.detach().cpu().clone()
                                              for k, v in net.state_dict().items()},
                                original=args.original, epoch=epoch,
                                val_acc=val_acc, seed=args.seed), checkpoint_path)
            print(f"epoch {epoch:02}/{args.epochs}: loss={row['train_loss']:.4f} "
                  f"train_acc={row['train_acc']:.4f} val_acc={val_acc:.4f} "
                  f"best_val={best_val_acc:.4f} time={time.perf_counter()-epoch_start:.1f}s",
                  flush=True)
            (args.output_dir / "history.json").write_text(
                json.dumps(history, indent=2), encoding="utf-8")
            draw_history(fig, axes, history)
            fig.savefig(args.output_dir / "training_curve.png", dpi=160)
            if not args.no_show:
                plt.pause(0.01)
        print(f"训练耗时 {time.perf_counter()-start:.1f}s；最佳验证轮次 {best_epoch}",
              flush=True)

    # 测试集不参与梯度更新、学习率调节或最佳轮次选择。
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    net = LeNet(original=saved["original"]).to(device)
    net.load_state_dict(saved["model_state"])
    test_loss, test_acc = evaluate(net, test_loader, device)
    result = dict(test_acc=test_acc, test_loss=test_loss,
                  best_epoch=saved["epoch"], validation_acc=saved["val_acc"],
                  original=saved["original"], seed=saved["seed"], device=str(device),
                  train_examples=54000, validation_examples=6000, test_examples=10000)
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print("FINAL RESULT " + json.dumps(result), flush=True)
    print(f"最佳模型：{checkpoint_path}", flush=True)
    if not args.evaluate:
        fig.suptitle(f"LeNet / Fashion-MNIST | final test accuracy: {test_acc:.2%}")
        fig.savefig(args.output_dir / "training_curve.png", dpi=160)
        if not args.no_show:
            plt.ioff()
            plt.show(block=True)


if __name__ == "__main__":
    main()
