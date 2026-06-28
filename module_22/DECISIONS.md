# Module 22 — DECISIONS.md

## Decision 1: nn.Sequential over custom nn.Module subclass for the layer stack
**Decision:** Store layers inside `nn.Sequential` within DetectiveMLP rather than defining forward() manually with explicit layer calls.
**Why:** nn.Sequential is idiomatic for linear feed-forward architectures. It keeps the architecture definition in one place (the constructor) and avoids writing repetitive `x = self.layer1(x); x = self.relu1(x); ...` chains. DetectiveMLP is still a proper nn.Module subclass for inheritance and parameter management.
**Trade-off:** nn.Sequential is inflexible — skip connections (ResNets) or multi-input architectures require manual forward(). For M24 (CNN) we switch to explicit forward() calls.

## Decision 2: SGD with momentum, not Adam
**Decision:** Use `optim.SGD(momentum=0.9)` rather than `optim.Adam`.
**Why:** Adam is the modern default (Module 23+ will use it), but SGD+momentum is conceptually closer to the gradient descent students wrote from scratch. Adding momentum here bridges the gap: momentum is just an exponential moving average of past gradients — a simple extension of the M18 update rule.
**Trade-off:** Adam converges faster in practice; SGD+momentum may need more epochs. Acceptable for a 150-sample dataset.

## Decision 3: weight_decay passed to optimizer, not manually added
**Decision:** Use PyTorch's built-in `weight_decay` parameter in SGD rather than manually adding `lambda * W` to gradients.
**Why:** This is how L2 regularisation is used in production PyTorch code. Students can verify it's the same operation as M21's `dl.dW += l2_lambda * dl.W` by reading the PyTorch source — optim.SGD's weight_decay does exactly that multiplication before the gradient step.
**Trade-off:** The mechanics are hidden — DECISIONS.md explicitly connects it to M21's implementation.

## Decision 4: torch.no_grad() in eval_epoch, not detach()
**Decision:** Use `with torch.no_grad():` context manager during validation, not `.detach()` on tensors.
**Why:** `torch.no_grad()` disables the entire autograd engine for the block — no computation graph is built, which saves memory proportional to the number of layers. `.detach()` only disconnects a specific tensor from the graph; the rest of the graph is still built. For validation (no backward pass needed), `no_grad()` is both correct and more efficient.
**Trade-off:** None — `no_grad()` is universally correct for inference.
