"""module_26/finetune_advanced.py — Advanced Transfer Learning Techniques.

Covers:
  - Progressive unfreezing (unfreeze layer groups one at a time)
  - Layer-wise learning rate decay (discriminative fine-tuning)
  - Domain adaptation concepts
  - Freeze schedule visualisation

Run standalone: python finetune_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_SEED = 42
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
PLOTS_DIR   = "plots"
EPOCHS      = 6   # 3 frozen + 3 unfrozen
BATCH_SIZE  = 32
torch.manual_seed(RANDOM_SEED)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PROGRESSIVE UNFREEZING
# ─────────────────────────────────────────────────────────────────────────────

def get_resnet_layer_groups(model: nn.Module) -> list:
    """Return ResNet-18 parameter groups from head → stem.

    Progressive unfreezing unfreezes groups from head (last) to stem (first).
    This is the opposite of training from scratch: you first train the head
    (which needs the most adjustment) then gradually unfreeze earlier layers.

    Args:
        model: ResNet-18 with custom fc

    Returns:
        List of (group_name, params) tuples, head first
    """
    return [
        ("head",   list(model.fc.parameters())),
        ("layer4", list(model.layer4.parameters())),
        ("layer3", list(model.layer3.parameters())),
        ("layer2", list(model.layer2.parameters())),
        ("layer1", list(model.layer1.parameters())),
        ("stem",   list(model.conv1.parameters()) + list(model.bn1.parameters())),
    ]


def freeze_all_except_head(model: nn.Module):
    """Freeze all parameters except the final fc layer."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")


def unfreeze_group(params: list):
    """Unfreeze a list of parameters."""
    for p in params:
        p.requires_grad = True


def count_trainable(model: nn.Module) -> tuple:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def demo_progressive_unfreeze():
    """Demonstrate the progressive unfreezing schedule."""
    print("── 1. Progressive Unfreezing Schedule ──────────────")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)

    groups = get_resnet_layer_groups(model)
    freeze_all_except_head(model)

    print(f"  {'Step':<6} {'Unfrozen group':<12} {'Trainable':>12} {'%':>8}")
    trainable, total = count_trainable(model)
    print(f"  {'0':<6} {'head only':<12} {trainable:>12,} {100*trainable/total:>7.1f}%")

    for step, (name, params) in enumerate(groups[1:], start=1):   # skip head (already unfrozen)
        unfreeze_group(params)
        trainable, total = count_trainable(model)
        print(f"  {step:<6} {'+'+name:<12} {trainable:>12,} {100*trainable/total:>7.1f}%")

    print("\n  Why progressive unfreezing?")
    print("  Early layers of ImageNet models are generic (edges, textures).")
    print("  Training them too early disrupts well-learned features before")
    print("  the task-specific head has converged — causing catastrophic forgetting.")
    print("  Start with just the head, let it converge, then open up earlier layers.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. DISCRIMINATIVE FINE-TUNING (layer-wise LR decay)
# ─────────────────────────────────────────────────────────────────────────────

def build_discriminative_optimizer(model: nn.Module,
                                    base_lr: float = 1e-3,
                                    decay: float = 0.3) -> optim.Optimizer:
    """Build Adam optimizer with exponentially decayed LR per layer group.

    Discriminative fine-tuning (Howard & Ruder, ULMFiT 2018):
    - Head layer gets base_lr
    - Each earlier group gets base_lr * decay^depth
    - Earlier (more generic) layers get much smaller LRs

    Args:
        model:   ResNet-18 with custom head
        base_lr: learning rate for the head (task-specific) layers
        decay:   multiplicative LR factor per group going toward stem

    Returns:
        Adam optimizer with per-group learning rates
    """
    groups = get_resnet_layer_groups(model)
    param_groups = []
    for depth, (name, params) in enumerate(groups):
        lr = base_lr * (decay ** depth)
        if params:   # skip empty groups
            param_groups.append({"params": params, "lr": lr, "name": name})

    # Print the schedule
    print(f"  {'Layer group':<12} {'LR':>12}")
    for pg in param_groups:
        print(f"  {pg['name']:<12} {pg['lr']:>12.2e}")

    return optim.Adam(param_groups)


def demo_discriminative_lr():
    """Show discriminative learning rate schedule."""
    print("\n── 2. Discriminative Learning Rate Decay ────────────")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)

    # First unfreeze all so all groups appear in optimizer
    for p in model.parameters():
        p.requires_grad = True

    optimizer = build_discriminative_optimizer(model, base_lr=1e-3, decay=0.3)

    # Visualise
    names = [pg["name"] for pg in optimizer.param_groups]
    lrs   = [pg["lr"]   for pg in optimizer.param_groups]

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    bars = ax.barh(names, lrs, color="#58a6ff", edgecolor="#30363d")
    for bar, lr in zip(bars, lrs):
        ax.text(bar.get_width() + max(lrs)*0.01, bar.get_y() + bar.get_height()/2,
                f"{lr:.2e}", va="center", color="#e6edf3", fontsize=9)
    ax.set_xlabel("Learning Rate (log scale)", color="#e6edf3")
    ax.set_xscale("log")
    ax.set_title("Discriminative Fine-Tuning: LR per Layer Group", color="#e6edf3")
    ax.tick_params(colors="#e6edf3")
    ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/discriminative_lr.png", dpi=120,
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {PLOTS_DIR}/discriminative_lr.png")
    print("\n  Stem layer LR is ~300× smaller than head.")
    print("  Generic low-level features barely move; head adapts aggressively.")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROGRESSIVE TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def get_small_cifar10(n: int = 500) -> tuple:
    """Load a small CIFAR-10 subset for a fast demo.

    Args:
        n: number of training examples per class

    Returns:
        (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=transform)
    test_ds  = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform)

    # Use only a small subset to keep the demo fast
    idx = list(range(n))
    train_loader = DataLoader(Subset(train_ds, idx), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    test_loader  = DataLoader(Subset(test_ds, list(range(200))),
                               batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            preds    = model(X.to(device)).argmax(1)
            correct += (preds == y.to(device)).sum().item()
            total   += len(y)
    return correct / total


def demo_progressive_training():
    """Run a short training loop with progressive unfreezing mid-way."""
    print("\n── 3. Progressive Fine-Tuning Training Loop ─────────")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    train_loader, test_loader = get_small_cifar10(n=400)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(512, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    groups    = get_resnet_layer_groups(model)

    # Phase 1: train head only (3 epochs)
    freeze_all_except_head(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3)
    print(f"\n  Phase 1: Head only (3 epochs)")
    print(f"  {'Epoch':>6} {'Loss':>8} {'Val Acc':>10}")
    for ep in range(1, 4):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc  = eval_accuracy(model, test_loader, device)
        print(f"  {ep:>6} {loss:>8.4f} {acc:>10.4f}")

    # Phase 2: unfreeze layer4 and lower LR (3 epochs)
    unfreeze_group(groups[1][1])   # layer4
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-4)
    print(f"\n  Phase 2: + layer4 unfrozen (3 epochs, lr=1e-4)")
    print(f"  {'Epoch':>6} {'Loss':>8} {'Val Acc':>10}")
    for ep in range(1, 4):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc  = eval_accuracy(model, test_loader, device)
        print(f"  {ep:>6} {loss:>8.4f} {acc:>10.4f}")

    print("\n  Progressive unfreezing prevents the early generic features from")
    print("  being destroyed before the head learns the new task vocabulary.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. DOMAIN ADAPTATION CONCEPTS
# ─────────────────────────────────────────────────────────────────────────────

def demo_domain_adaptation():
    """Explain domain adaptation concepts with pseudocode patterns."""
    print("\n── 4. Domain Adaptation ────────────────────────────")
    print("""
  PROBLEM: Train on source domain (e.g. natural photos), deploy on
  target domain (e.g. medical scans, satellite imagery, manga).

  STRATEGIES:
  ───────────────────────────────────────────────────────────────
  1. Fine-tune with target data (if you have labels)
     → Simplest. Use discriminative LRs + progressive unfreezing.

  2. Few-shot fine-tune (5-50 labeled target examples)
     → Freeze backbone, train only classification head.
     → Works because ImageNet features transfer surprisingly well.

  3. Domain-Adversarial Neural Network (DANN)
     → Add gradient reversal layer between feature extractor and
       domain classifier. Feature extractor learns to be domain-agnostic.
     → Use when: unlabeled target data available, labeled source data only.

  4. Self-supervised pre-training on target domain (SimCLR, MoCo)
     → Pre-train on unlabeled target images with contrastive loss.
     → Then fine-tune with small labeled set.

  PRACTICAL CHECKLIST:
  ───────────────────────────────────────────────────────────────
  ✓ Do source and target images have similar statistics?
    → Check mean/std/histogram. If very different, consider:
       a) normalise using target domain statistics
       b) domain-specific augmentation (e.g. blur for foggy images)

  ✓ How many labeled target examples do you have?
    → <100:   freeze backbone, fine-tune head only
    → 100-1k: progressive unfreezing + discriminative LRs
    → >1k:    full fine-tune with lower LR

  ✓ Is the task definition the same?
    → New classes not in source? → replace head, initialise randomly.
    → Same classes but different appearance? → keep head, fine-tune.
  """)


def main():
    print("=" * 54)
    print("MODULE 26 — Advanced Transfer Learning")
    print("=" * 54)
    demo_progressive_unfreeze()
    demo_discriminative_lr()
    demo_progressive_training()
    demo_domain_adaptation()
    print("\nDone.")


if __name__ == "__main__":
    main()
