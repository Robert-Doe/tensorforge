"""module_24/resnet_block.py — Residual Connections and ResNet blocks.

Covers:
  - Residual (skip) connection from scratch
  - Why residuals solve vanishing gradients
  - Basic ResNet block (with and without projection shortcut)
  - Bottleneck block (used in ResNet-50+)
  - Small ResNet trained on CIFAR-10 style data

Run standalone: python resnet_block.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

RANDOM_SEED = 42
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
EPOCHS      = 3      # short demo
BATCH_SIZE  = 64
LR          = 1e-3
torch.manual_seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC RESIDUAL BLOCK
# ─────────────────────────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3×3 convs with a skip connection.

    Output: F(x) + x  where F = Conv-BN-ReLU-Conv-BN
    The shortcut adds the input directly, bypassing the two conv layers.

    If input and output channels differ (stride>1 or channel change),
    a 1×1 projection conv is used to match dimensions.

    Args:
        in_channels:  number of input channels
        out_channels: number of output channels
        stride:       stride for the first conv (>1 = spatial downsampling)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Projection shortcut: match dimensions when stride or channels differ
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """F(x) + shortcut(x).

        Args:
            x: input tensor

        Returns:
            residual output after ReLU
        """
        out = F.relu(self.bn1(self.conv1(x)))   # first conv path
        out = self.bn2(self.conv2(out))          # second conv path (no ReLU yet)
        out = out + self.shortcut(x)             # add skip connection
        out = F.relu(out)                        # ReLU after addition
        return out


class BottleneckBlock(nn.Module):
    """Bottleneck block used in ResNet-50/101/152.

    1×1 (reduce channels) → 3×3 (spatial conv) → 1×1 (restore channels)
    Much fewer parameters than equivalent BasicBlock at same output channels.

    expansion=4: output channels = in_channels * 4

    Args:
        in_channels:  input channels
        bottleneck_ch: channels in the narrow 3×3 layer (= out_channels // 4)
        stride:        spatial stride for 3×3 conv
    """

    expansion = 4

    def __init__(self, in_channels: int, bottleneck_ch: int, stride: int = 1):
        super().__init__()
        out_channels = bottleneck_ch * self.expansion

        self.conv1 = nn.Conv2d(in_channels, bottleneck_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(bottleneck_ch)
        self.conv2 = nn.Conv2d(bottleneck_ch, bottleneck_ch, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(bottleneck_ch)
        self.conv3 = nn.Conv2d(bottleneck_ch, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))   # 1×1 reduce
        out = F.relu(self.bn2(self.conv2(out))) # 3×3 spatial
        out = self.bn3(self.conv3(out))          # 1×1 restore
        out = F.relu(out + self.shortcut(x))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. SMALL RESNET for MNIST
# ─────────────────────────────────────────────────────────────────────────────

class SmallResNet(nn.Module):
    """Compact ResNet (4 basic blocks) for MNIST digit classification.

    Architecture: initial conv → 2 BasicBlocks (×2) → avgpool → fc
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        # Two groups of blocks
        self.layer1 = nn.Sequential(
            BasicBlock(16, 16),
            BasicBlock(16, 16),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),   # 28→14
            BasicBlock(32, 32),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pooling → (B, 32, 1, 1)
        self.fc   = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x).view(x.size(0), -1)   # (B, 32)
        return self.fc(x)   # raw logits — use CrossEntropyLoss


# ─────────────────────────────────────────────────────────────────────────────
# 3. WHY RESIDUALS SOLVE VANISHING GRADIENTS
# ─────────────────────────────────────────────────────────────────────────────

def demo_vanishing_gradient():
    """Show gradient norms in plain network vs residual network."""
    print("── 1. Vanishing Gradients: Plain vs Residual ────────")
    torch.manual_seed(RANDOM_SEED)

    depth = 20   # 20 conv layers — very deep

    # Plain network (no skip connections)
    plain_layers = []
    for _ in range(depth):
        plain_layers += [nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()]
    plain_net = nn.Sequential(*plain_layers)

    # Residual network (BasicBlock every 2 layers)
    res_blocks = nn.Sequential(*[BasicBlock(16, 16) for _ in range(depth // 2)])

    x = torch.randn(4, 16, 28, 28, requires_grad=True)
    criterion = nn.MSELoss()

    print(f"  Depth: {depth} layers")
    for name, net in [("Plain CNN", plain_net), ("ResNet", res_blocks)]:
        out  = net(x)
        loss = criterion(out, torch.zeros_like(out))
        loss.backward()
        grad_norm = x.grad.norm().item()
        x.grad.zero_()
        print(f"  {name:<15}: gradient norm at input = {grad_norm:.6f}")

    print("  Plain: gradient may vanish (near 0) — early layers get no useful signal.")
    print("  ResNet: gradient flows through the identity shortcut → healthy norm.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PARAMETER COUNT COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def demo_block_params():
    """Compare parameter counts: BasicBlock vs BottleneckBlock."""
    print("\n── 2. Parameter Count: BasicBlock vs Bottleneck ────")
    basic  = BasicBlock(64, 64)
    bottle = BottleneckBlock(64, 16)   # 16 * 4 = 64 output channels

    p_basic  = sum(p.numel() for p in basic.parameters())
    p_bottle = sum(p.numel() for p in bottle.parameters())

    print(f"  BasicBlock(64→64):         {p_basic:>8,} params")
    print(f"  BottleneckBlock(64→16→64): {p_bottle:>8,} params")
    print(f"  Reduction: {p_basic/p_bottle:.1f}×")
    print("  Bottleneck achieves same depth with fewer parameters.")


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAIN SmallResNet on MNIST (short demo)
# ─────────────────────────────────────────────────────────────────────────────

def demo_train():
    """Train SmallResNet for 3 epochs to confirm it works."""
    print("\n── 3. SmallResNet Training Demo (3 epochs) ─────────")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    g         = torch.Generator().manual_seed(RANDOM_SEED)
    train_ld  = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_ld   = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SmallResNet(num_classes=10).to(device)
    optim_ = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  SmallResNet parameters: {total_params:,}")

    print(f"  {'Epoch':>5}  {'Train Acc':>10}  {'Test Acc':>10}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_correct = tr_total = 0
        for X, y in train_ld:
            X, y = X.to(device), y.to(device)
            optim_.zero_grad()
            out  = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            optim_.step()
            tr_correct += (out.argmax(1) == y).sum().item()
            tr_total   += len(y)

        model.eval()
        te_correct = te_total = 0
        with torch.no_grad():
            for X, y in test_ld:
                out = model(X.to(device))
                te_correct += (out.argmax(1) == y.to(device)).sum().item()
                te_total   += len(y)
        print(f"  {epoch:>5}  {tr_correct/tr_total:>10.4f}  {te_correct/te_total:>10.4f}")


def main():
    print("=" * 54)
    print("MODULE 24 — Residual Connections & ResNet Blocks")
    print("=" * 54)
    demo_vanishing_gradient()
    demo_block_params()
    demo_train()
    print("\nDone.")


if __name__ == "__main__":
    main()
