# tensorforge

Frameworks like scikit-learn and PyTorch are excellent at hiding the math that makes them work. That's great when you're shipping and terrible when you're trying to learn. I built tensorforge on the opposite bet: implement the core algorithm yourself first, gradient descent, backprop, convolution, k-means, attention, in raw NumPy, and only graduate to the production library once you've actually felt the mechanism with nothing standing between you and it. Every module is real, runnable code (`main.py` plus a `*_advanced.py`), a `DECISIONS.md` explaining the pedagogical and engineering trade-offs behind it, and a styled `tutorial.html` walkthrough. This runs longer and goes deeper than a typical intro-to-ML course, 28 modules, meant to be run end to end, not skimmed on a lunch break.

## Module map

| # | Module | Topic |
|---|---|---|
| 01 | `module_01` | NumPy fundamentals: vectorized computation, `default_rng`, why arrays beat lists |
| 02 | `module_02` | Pandas fundamentals: data wrangling, `pd.cut` bucketing |
| 03 | `module_03` | Data visualization: six chart types over one shared dataset, rolling averages |
| 04 | `module_04` | Linear regression from scratch: leakage-free train/test splitting, synthetic ground truth |
| 05 | `module_05` | Logistic regression: stratified splits and feature scaling for imbalanced fraud detection |
| 06 | `module_06` | K-Nearest Neighbors: distance-weighted voting, odd-K tie avoidance |
| 07 | `module_07` | Decision trees: Gini impurity splitting, overfitting via fully-grown trees |
| 08 | `module_08` | Naive Bayes: Gaussian, Multinomial, and Bernoulli variants, online learning via `partial_fit` |
| 09 | `module_09` | Advanced model evaluation: bootstrap confidence intervals, learning and validation curves, Cohen's Kappa, McNemar's test |
| 10 | `module_10` | Transition to scikit-learn: reusing the shared data pipeline, production-grade estimators |
| 11 | `module_11` | Support Vector Machines: pipelines with `StandardScaler`, Platt-scaled probabilities |
| 12 | `module_12` | Ensembles: Random Forest parallelism vs. sequential Gradient Boosting |
| 13 | `module_13` | K-Means clustering from scratch: random initialization, explicit inertia computation |
| 14 | `module_14` | PCA: symmetric eigendecomposition (`eigh`), leakage-free fit/transform |
| 15 | `module_15` | Feature engineering: one-hot vs. ordinal encoding, `ColumnTransformer` pipelines |
| 16 | `module_16` | The neuron: showing that a single neuron is mathematically a logistic regression unit |
| 17 | `module_17` | Backpropagation from scratch: a 2-layer net, He/Xavier initialization |
| 18 | `module_18` | Building an MLP framework: layer abstraction with `forward()`/`backward()`, solving XOR |
| 19 | `module_19` | Activation functions and the vanishing gradient problem, visualized on a log scale |
| 20 | `module_20` | Mini-batch training: a dedicated trainer class, train/val/test three-way splits |
| 21 | `module_21` | Regularization: inverted dropout and L2 weight decay implemented in the backward pass |
| 22 | `module_22` | Introduction to PyTorch: `nn.Sequential`, SGD with momentum, bridging from scratch code |
| 23 | `module_23` | Convolution fundamentals: hand-crafted Sobel, blur, and sharpen kernels, forward pass only |
| 24 | `module_24` | CNN classifier in PyTorch: `log_softmax` plus `NLLLoss`, Adam optimizer |
| 25 | `module_25` | BatchNorm and Dropout: Conv-BN-ReLU ordering, a three-way ablation study |
| 26 | `module_26` | Transfer learning: fine-tuning on CIFAR-10, frozen vs. full fine-tune vs. from-scratch ablation |
| 27 | `module_27` | Recurrent networks: a vanilla RNN in NumPy, then LSTM and attention in PyTorch |
| 28 | `module_28` | End-to-end pipeline and serving: leakage-free imputation, a full production-style workflow |

## Tech stack

Python 3, NumPy, pandas, and matplotlib carry the from-scratch modules (01 through 21, and 23). scikit-learn handles the production-grade classical ML modules (10 through 15, and 28). PyTorch, with torchvision, covers the neural networks, CNNs, transfer learning, and RNN/LSTM work (modules 22, and 24 through 27).

## Where this stands

Done. 28 modules, each with a runnable `main.py`, an advanced companion script, a `DECISIONS.md` design log, and an HTML tutorial. The course runs in strict order on purpose, every module reuses data and abstractions built in earlier ones. The neuron in module 16, for instance, is the same logistic regression from module 05 wearing different notation, and the MLP layer abstraction from module 18 carries straight through into the PyTorch modules.

## How to run it

Each module is self-contained enough to run on its own once dependencies are installed:

```bash
python3 -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install numpy pandas matplotlib scikit-learn torch torchvision

python module_01/main.py
```

Later modules (10 and up) ship their own `requirements.txt`, install per module if you're jumping around instead of working straight through:

```bash
pip install -r module_10/requirements.txt
```

Start at `module_01/tutorial.html` and work through in numeric order.
