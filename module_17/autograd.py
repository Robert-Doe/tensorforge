"""module_17/autograd.py — Automatic Differentiation from scratch.

Covers:
  - Scalar computational graph (Value class with backward())
  - Numerical gradient checking (finite differences vs analytical)
  - Forward-mode vs reverse-mode AD conceptually
  - Why PyTorch uses reverse-mode (efficient for many inputs, few outputs)

Run standalone: python autograd.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# 1. SCALAR COMPUTATIONAL GRAPH — minigrad-style
# ─────────────────────────────────────────────────────────────────────────────

class Value:
    """Scalar value with automatic differentiation via reverse-mode AD.

    Each Value stores:
      - data:     the scalar value
      - grad:     accumulated gradient dL/d(self) after backward()
      - _backward: closure that propagates gradient to children
      - _prev:    set of child Values (the inputs to this operation)

    Supports: +, *, **, tanh, relu, exp, log.
    After constructing a computation graph, call .backward() on the output
    to fill in .grad on all nodes via the chain rule in reverse order.
    """

    def __init__(self, data: float, _children=(), _op='', label=''):
        self.data      = float(data)
        self.grad      = 0.0           # dL/d(self), accumulated
        self._backward = lambda: None  # default: leaf node, no children to update
        self._prev     = set(_children)
        self._op       = _op
        self.label     = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ── Arithmetic operations ─────────────────────────────────────────────────

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, so grad flows through unchanged
            self.grad  += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out   = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))
        out = Value(self.data ** exponent, (self,), f'**{exponent}')

        def _backward():
            # d(x^n)/dx = n * x^{n-1}
            self.grad += exponent * (self.data ** (exponent - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * (other ** -1)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return other + (-self)

    def tanh(self):
        t   = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # d(tanh(x))/dx = 1 - tanh²(x)
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        e   = math.exp(self.data)
        out = Value(e, (self,), 'exp')

        def _backward():
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data + 1e-8), (self,), 'log')

        def _backward():
            self.grad += (1.0 / (self.data + 1e-8)) * out.grad

        out._backward = _backward
        return out

    # ── Reverse-mode backward pass ────────────────────────────────────────────

    def backward(self):
        """Run reverse-mode AD on the entire computation graph.

        Uses topological sort to process nodes in dependency order:
        output first, inputs last. Gradient starts at 1.0 (dL/dL = 1).
        """
        topo   = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0   # seed gradient

        for v in reversed(topo):
            v._backward()


# ─────────────────────────────────────────────────────────────────────────────
# 2. DEMO — trace through a simple MLP forward + backward
# ─────────────────────────────────────────────────────────────────────────────

def demo_scalar_autograd():
    """Build a tiny 2-input neuron and verify gradients analytically."""
    print("── 1. Scalar Autograd — Single Neuron ──────────────")
    # Inputs and weights
    x1 = Value(2.0,  label='x1')
    x2 = Value(0.0,  label='x2')
    w1 = Value(-3.0, label='w1')
    w2 = Value(1.0,  label='w2')
    b  = Value(6.8813735870195430, label='b')

    # Forward: tanh(x1*w1 + x2*w2 + b)
    x1w1 = x1 * w1;  x1w1.label = 'x1*w1'
    x2w2 = x2 * w2;  x2w2.label = 'x2*w2'
    xw   = x1w1 + x2w2; xw.label = 'xw'
    n    = xw + b;   n.label = 'n'
    o    = n.tanh(); o.label = 'o'

    print(f"  Output: {o.data:.6f}  (should be ≈ 0.707107)")

    o.backward()

    print(f"  do/dx1 = {x1.grad:.6f}  (analytical: w1*(1-tanh²(n)) = {w1.data*(1-o.data**2):.6f})")
    print(f"  do/dx2 = {x2.grad:.6f}  (analytical: w2*(1-tanh²(n)) = {w2.data*(1-o.data**2):.6f})")
    print(f"  do/dw1 = {w1.grad:.6f}  (analytical: x1*(1-tanh²(n)) = {x1.data*(1-o.data**2):.6f})")
    print(f"  do/db  = {b.grad:.6f}   (analytical: 1*(1-tanh²(n)) = {1*(1-o.data**2):.6f})")


# ─────────────────────────────────────────────────────────────────────────────
# 3. NUMERICAL GRADIENT CHECK
# ─────────────────────────────────────────────────────────────────────────────

def numerical_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Compute gradient of f at x via central finite differences.

    f'(x_i) ≈ [f(x + eps*e_i) - f(x - eps*e_i)] / (2*eps)
    Central differences have O(eps²) error vs O(eps) for forward differences.

    Args:
        f:   scalar-valued function of a numpy array
        x:   point at which to compute gradient
        eps: finite difference step size

    Returns:
        numerical gradient, same shape as x
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus        = x.copy(); x_plus[i]  += eps
        x_minus       = x.copy(); x_minus[i] -= eps
        grad[i]       = (f(x_plus) - f(x_minus)) / (2 * eps)
    return grad


def demo_gradient_check():
    """Verify our manual backprop implementation against numerical gradients."""
    print("\n── 2. Numerical Gradient Check ─────────────────────")
    rng = np.random.default_rng(42)

    # Simple function: L2 loss for a 1-layer network
    def forward_np(params, X, y):
        """params = [w0, w1, w2, b]"""
        w = params[:3]
        b = params[3]
        pred   = np.tanh(X @ w + b)
        loss   = ((pred - y) ** 2).mean()
        return loss

    X     = rng.standard_normal((10, 3))
    y     = rng.integers(0, 2, size=10).astype(float)
    theta = rng.standard_normal(4)

    f       = lambda p: forward_np(p, X, y)
    num_grad = numerical_gradient(f, theta)

    # Analytical gradient via backprop (compute manually)
    w = theta[:3]; b = theta[3]
    a = np.tanh(X @ w + b)
    delta = 2 * (a - y) / len(y) * (1 - a**2)
    ana_grad_w = X.T @ delta
    ana_grad_b = delta.sum()
    ana_grad   = np.append(ana_grad_w, ana_grad_b)

    print(f"  {'Param':<10} {'Numerical':>12}  {'Analytical':>12}  {'Rel error':>12}")
    for i, (num, ana) in enumerate(zip(num_grad, ana_grad)):
        rel_err = abs(num - ana) / (abs(ana) + 1e-8)
        name = f"w{i}" if i < 3 else "b"
        print(f"  {name:<10} {num:>12.6f}  {ana:>12.6f}  {rel_err:>12.2e}")

    max_rel_err = np.max(np.abs(num_grad - ana_grad) / (np.abs(ana_grad) + 1e-8))
    print(f"\n  Max relative error: {max_rel_err:.2e}  {'✓ PASS' if max_rel_err < 1e-4 else '✗ FAIL'}")
    print("  Relative error < 1e-4 confirms the analytical gradient is correct.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. FORWARD vs REVERSE MODE AD — conceptual comparison
# ─────────────────────────────────────────────────────────────────────────────

def demo_ad_modes():
    """Explain forward vs reverse mode with complexity analysis."""
    print("\n── 3. Forward vs Reverse Mode AD ───────────────────")
    print("""
  FORWARD MODE (Jvp — Jacobian-vector product):
  • Propagates derivatives forward from INPUTS to OUTPUTS.
  • Cost: O(n_inputs) passes to compute all partial derivatives.
  • Efficient when: n_inputs << n_outputs (e.g. ODE sensitivity).
  • PyTorch supports this via torch.func.jvp.

  REVERSE MODE (vjp — vector-Jacobian product) = BACKPROP:
  • Propagates gradient backward from OUTPUTS to INPUTS.
  • Cost: ONE forward pass + ONE backward pass (≈2× forward cost).
  • Efficient when: n_inputs >> n_outputs (typical in ML: millions of
    parameters → 1 scalar loss).
  • This is what model.backward() does in PyTorch.

  WHY REVERSE MODE WINS FOR DEEP LEARNING:
  • A typical network: 10M parameters (inputs), 1 loss value (output).
  • Forward mode: 10M separate passes → prohibitive.
  • Reverse mode: 1 backward pass → computes all 10M gradients at once.

  Memory tradeoff:
  • Reverse mode must STORE all intermediate activations during the
    forward pass (for use in the backward pass). This is why large
    networks need a lot of GPU memory.
  • Gradient checkpointing trades compute for memory: recompute
    activations during backward instead of storing them.
""")


def main():
    print("=" * 54)
    print("MODULE 17 — Automatic Differentiation")
    print("=" * 54)
    demo_scalar_autograd()
    demo_gradient_check()
    demo_ad_modes()
    print("Done.")


if __name__ == "__main__":
    main()
