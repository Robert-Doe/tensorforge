"""module_22/pytorch_advanced.py — Advanced PyTorch patterns.

Covers:
  - Custom Dataset and DataLoader
  - Model checkpointing (save/load state_dict)
  - Learning rate scheduling (torch.optim.lr_scheduler)
  - Forward hooks for activation inspection
  - torch.compile (PyTorch 2.0 speedup)
  - Mixed precision training concept (AMP)
  - Parameter groups with different learning rates

Run standalone: python pytorch_advanced.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
MODELS_DIR  = "saved_models"
torch.manual_seed(RANDOM_SEED)
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CUSTOM DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TabularDataset(Dataset):
    """Custom Dataset for tabular (numpy array) data.

    Wraps numpy arrays in a torch Dataset. The DataLoader will call
    __getitem__ to retrieve individual samples and batch them.

    Args:
        X: feature array (n_samples, n_features)
        y: label array (n_samples,)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def demo_custom_dataset():
    """Show how to build and use a custom Dataset."""
    print("── 1. Custom Dataset & DataLoader ──────────────────")
    X, y = make_classification(n_samples=1000, n_features=20,
                               random_state=RANDOM_SEED)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=RANDOM_SEED)

    train_ds = TabularDataset(X_tr, y_tr)
    test_ds  = TabularDataset(X_te, y_te)

    g = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              generator=g, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False,
                              num_workers=0)

    X_batch, y_batch = next(iter(train_loader))
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Batch shapes: X={X_batch.shape}  y={y_batch.shape}")
    print(f"  dtype: X={X_batch.dtype}  y={y_batch.dtype}")
    return train_loader, test_loader, X_tr.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL WITH PARAMETER GROUPS (different LRs per layer)
# ─────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Simple MLP for tabular classification."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu   = nn.ReLU()
        self.drop   = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.drop(x)
        x = self.relu(self.layer2(x))
        return torch.sigmoid(self.output(x))


def demo_parameter_groups(input_dim: int):
    """Different learning rates for early vs late layers."""
    print("\n── 2. Parameter Groups (layer-wise LR) ─────────────")
    model = MLP(input_dim)

    # Early layers get a lower LR (they change slowly once trained)
    # Output layer gets a higher LR (needs to adapt quickly)
    param_groups = [
        {"params": model.layer1.parameters(), "lr": 1e-4, "name": "layer1"},
        {"params": model.layer2.parameters(), "lr": 5e-4, "name": "layer2"},
        {"params": model.output.parameters(), "lr": 1e-3,  "name": "output"},
    ]
    optimizer = optim.Adam(param_groups)

    print("  Parameter groups:")
    for pg in optimizer.param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        print(f"    {pg['name']:<10}: lr={pg['lr']}  params={n_params:,}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRAINING WITH SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

def train_with_scheduler(model, train_loader, test_loader, device, epochs=15):
    """Train model with CosineAnnealingLR and print LR at each epoch."""
    print("\n── 3. Training with LR Scheduler ───────────────────")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    # Cosine annealing: LR goes from 1e-2 to ~0 over T_max epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    model = model.to(device)
    print(f"  {'Epoch':>5}  {'LR':>10}  {'Val Acc':>10}")

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()   # update LR after each epoch

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_b, y_b in test_loader:
                preds  = (model(X_b.to(device)) >= 0.5).float().cpu()
                correct += (preds.squeeze() == y_b).sum().item()
                total   += len(y_b)

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 3 == 1 or epoch == epochs:
            print(f"  {epoch:>5}  {current_lr:>10.6f}  {correct/total:>10.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. MODEL CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────

def demo_checkpointing(model, optimizer):
    """Save and load model checkpoint (state_dict)."""
    print("\n── 4. Model Checkpointing ──────────────────────────")
    checkpoint_path = f"{MODELS_DIR}/mlp_checkpoint.pt"

    # Save: state_dict contains all learnable parameter tensors
    # Save optimizer state too — important for resuming training with momentum
    torch.save({
        "epoch":               5,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":             0.921,
    }, checkpoint_path)
    print(f"  Saved checkpoint → {checkpoint_path}")

    # Load: must create model with same architecture first
    new_model = MLP(input_dim=20)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    new_model.load_state_dict(checkpoint["model_state_dict"])
    new_model.eval()
    print(f"  Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  Saved val_acc: {checkpoint['val_acc']}")

    # Verify loaded weights match
    for (n1, p1), (n2, p2) in zip(model.named_parameters(),
                                    new_model.named_parameters()):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"
    print("  Weight verification: ✓ identical")


# ─────────────────────────────────────────────────────────────────────────────
# 5. FORWARD HOOKS — inspect activations without modifying model
# ─────────────────────────────────────────────────────────────────────────────

def demo_hooks(model, train_loader, device):
    """Attach hooks to capture intermediate activations."""
    print("\n── 5. Forward Hooks (activation inspection) ────────")
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks on specific layers
    h1 = model.layer1.register_forward_hook(make_hook("layer1"))
    h2 = model.layer2.register_forward_hook(make_hook("layer2"))

    model.eval()
    X_batch, _ = next(iter(train_loader))
    with torch.no_grad():
        model(X_batch.to(device))

    for name, act in activations.items():
        print(f"  {name}: shape={act.shape}  "
              f"mean={act.mean().item():.4f}  std={act.std().item():.4f}")

    # Always remove hooks when done — they persist and can cause memory leaks
    h1.remove(); h2.remove()
    print("  Hooks removed.")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MIXED PRECISION CONCEPT
# ─────────────────────────────────────────────────────────────────────────────

def demo_mixed_precision_concept():
    """Explain AMP — Automatic Mixed Precision."""
    print("\n── 6. Mixed Precision Training (AMP) ───────────────")
    print("""
  float32 (standard):  32 bits per value  — slow, accurate
  float16 (half):      16 bits per value  — 2× faster on GPU, less memory
  bfloat16:            brain float 16     — better range than float16

  AMP automatically uses float16 for forward/backward passes
  and float32 for weight updates. The gradient scaler prevents
  float16 underflow (gradients becoming 0).

  PyTorch AMP usage:
  ─────────────────
  from torch.cuda.amp import autocast, GradScaler
  scaler = GradScaler()

  for X, y in loader:
      optimizer.zero_grad()
      with autocast():                   # ops inside use float16
          out  = model(X)
          loss = criterion(out, y)
      scaler.scale(loss).backward()      # scale loss to prevent underflow
      scaler.step(optimizer)             # unscale grads, then step
      scaler.update()                    # adjust scale factor

  Typical speedup on NVIDIA GPU: 1.5×–3×.
  Requires GPU that supports Tensor Cores (Volta/Turing/Ampere/Ada).
""")


def main():
    print("=" * 54)
    print("MODULE 22 — Advanced PyTorch Patterns")
    print("=" * 54)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_loader, test_loader, input_dim = demo_custom_dataset()
    model = demo_parameter_groups(input_dim)

    model = train_with_scheduler(model, train_loader, test_loader, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    demo_checkpointing(model, optimizer)
    demo_hooks(model, train_loader, device)
    demo_mixed_precision_concept()
    print("\nDone.")


if __name__ == "__main__":
    main()
