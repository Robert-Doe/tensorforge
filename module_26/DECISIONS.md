# Module 26 — DECISIONS.md

## Decision 1: CIFAR-10 instead of a truly "custom" dataset
**Decision:** Use torchvision's CIFAR-10 as the target dataset rather than a manually curated folder of images.
**Why:** A real custom dataset would require students to source, label, and organise their own images — a substantial side-quest that distracts from the transfer learning lesson. CIFAR-10 is a well-understood benchmark with 10 real-world classes (animals, vehicles) that clearly differs from ImageNet, making the transfer story credible. The API (`ImageFolder` vs `CIFAR10`) is identical in usage.
**Trade-off:** Students don't experience the full "I have 200 photos of my own" scenario. The tutorial explicitly calls this out and points to `torchvision.datasets.ImageFolder` for the real-world case.

## Decision 2: Three-way ablation (frozen / full fine-tune / from scratch)
**Decision:** Run three experiments and compare accuracy curves.
**Why:** The primary question in transfer learning is "how much does pretraining help?" A single experiment can't answer this — you need a baseline (from scratch) and a spectrum of fine-tuning strategies (head-only vs full). The ablation also shows that full fine-tuning with an inappropriate LR can perform worse than a frozen backbone.
**Trade-off:** Three experiments make the run ~3× longer on CPU. Documented in the tutorial with expected runtimes.

## Decision 3: Different learning rates for frozen vs fine-tuning
**Decision:** Use lr=1e-3 for the frozen case, lr=1e-4 for full fine-tuning.
**Why:** The pre-trained backbone weights are already near a good optimum. A standard lr=1e-3 would shift backbone weights too far in one epoch, destroying the learned representations. The rule of thumb is to use 10× lower LR for the backbone when fine-tuning. This is one of the most common transfer learning mistakes; making it explicit is pedagogically important.
**Trade-off:** Optimal LRs are dataset-dependent. The 10× rule is a heuristic, not a law — acknowledged in the tutorial.

## Decision 4: filter(lambda p: p.requires_grad, ...) in optimizer
**Decision:** Pass only trainable parameters to the optimizer rather than all model parameters.
**Why:** If frozen parameters are passed to Adam, Adam allocates momentum state for them even though their gradients are always zero — wasted memory and a source of confusion. Filtering to `requires_grad=True` parameters is correct and minimal.
**Trade-off:** If the caller later unfreezes parameters (e.g. for progressive unfreezing), the optimizer won't track them unless re-instantiated. Acceptable for a single-stage fine-tuning workflow.
