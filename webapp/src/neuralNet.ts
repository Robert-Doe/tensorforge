/**
 * neuralNet.ts
 *
 * A direct TypeScript port of module_17/backprop.py's `TwoLayerNet`:
 * a real 2-layer neural network (input -> ReLU hidden layer -> sigmoid
 * output) trained with backpropagation and full-batch gradient descent.
 *
 * Forward pass:
 *   Z1 = X @ W1 + b1      A1 = relu(Z1)
 *   Z2 = A1 @ W2 + b2     A2 = sigmoid(Z2)
 *
 * Backward pass (binary cross-entropy loss + sigmoid output simplifies
 * dL/dZ2 to A2 - y, the standard result used in the Python source):
 *   dZ2 = A2 - y
 *   dW2 = A1^T @ dZ2 / n         db2 = mean(dZ2)
 *   dA1 = dZ2 @ W2^T
 *   dZ1 = dA1 * reluDerivative(Z1)
 *   dW1 = X^T @ dZ1 / n          db1 = mean(dZ1)
 *
 * Every matrix here is a plain number[][] (row-major) with tiny hand-rolled
 * linear algebra helpers below, no numeric library, matching the "no
 * libraries" spirit of the from-scratch course module.
 */

export type Matrix = number[][];

function zeros(rows: number, cols: number): Matrix {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function matmul(a: Matrix, b: Matrix): Matrix {
  const n = a.length;
  const k = a[0].length;
  const m = b[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) {
    for (let p = 0; p < k; p++) {
      const aip = a[i][p];
      if (aip === 0) continue;
      const brow = b[p];
      const orow = out[i];
      for (let j = 0; j < m; j++) {
        orow[j] += aip * brow[j];
      }
    }
  }
  return out;
}

function transpose(a: Matrix): Matrix {
  const n = a.length;
  const m = a[0].length;
  const out = zeros(m, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) out[j][i] = a[i][j];
  }
  return out;
}

function addRowVec(a: Matrix, v: number[]): Matrix {
  return a.map((row) => row.map((val, j) => val + v[j]));
}

function elemwise(a: Matrix, f: (v: number) => number): Matrix {
  return a.map((row) => row.map(f));
}

function elemwiseMul(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((v, j) => v * b[i][j]));
}

function sub(a: Matrix, b: Matrix): Matrix {
  return a.map((row, i) => row.map((v, j) => v - b[i][j]));
}

function colMean(a: Matrix): number[] {
  const n = a.length;
  const m = a[0].length;
  const out = new Array(m).fill(0);
  for (const row of a) for (let j = 0; j < m; j++) out[j] += row[j] / n;
  return out;
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

function relu(z: number): number {
  return Math.max(0, z);
}

function reluDerivative(z: number): number {
  return z > 0 ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Small seeded RNG (mulberry32) + Box-Muller gaussian, so runs are reproducible.
// ---------------------------------------------------------------------------
export function makeRng(seed: number) {
  let s = seed | 0;
  return function rand(): number {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianFrom(rand: () => number): number {
  const u1 = Math.max(rand(), 1e-9);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export interface Sample {
  x: [number, number];
  label: 0 | 1;
}

/** Two interleaving crescent clusters (a "moons"-style, non-linearly-separable dataset). */
export function makeMoons(n = 160, seed = 7, noise = 0.15): Sample[] {
  const rand = makeRng(seed);
  const samples: Sample[] = [];
  const half = Math.floor(n / 2);
  for (let i = 0; i < half; i++) {
    const t = (i / half) * Math.PI;
    const x = Math.cos(t);
    const y = Math.sin(t);
    samples.push({
      x: [x + gaussianFrom(rand) * noise, y + gaussianFrom(rand) * noise],
      label: 0,
    });
  }
  for (let i = 0; i < n - half; i++) {
    const t = (i / half) * Math.PI;
    const x = 1 - Math.cos(t);
    const y = 1 - Math.sin(t) - 0.5;
    samples.push({
      x: [x + gaussianFrom(rand) * noise, y + gaussianFrom(rand) * noise],
      label: 1,
    });
  }
  return samples;
}

/** Classic XOR pattern: four Gaussian blobs at the corners, diagonal pairs share a label. */
export function makeXor(n = 160, seed = 7, spread = 0.28): Sample[] {
  const rand = makeRng(seed);
  const samples: Sample[] = [];
  const centers: Array<{ c: [number, number]; label: 0 | 1 }> = [
    { c: [-1, -1], label: 0 },
    { c: [1, 1], label: 0 },
    { c: [-1, 1], label: 1 },
    { c: [1, -1], label: 1 },
  ];
  for (let i = 0; i < n; i++) {
    const { c, label } = centers[i % centers.length];
    samples.push({
      x: [c[0] + gaussianFrom(rand) * spread, c[1] + gaussianFrom(rand) * spread],
      label,
    });
  }
  return samples;
}

export class TwoLayerNet {
  W1: Matrix;
  b1: number[];
  W2: Matrix;
  b2: number[];
  lr: number;
  hiddenSize: number;
  step_ = 0;
  lossHistory: number[] = [];

  constructor(nFeatures = 2, hiddenSize = 8, lr = 0.5, seed = 42) {
    this.hiddenSize = hiddenSize;
    this.lr = lr;
    const rand = makeRng(seed);

    // He initialization for the ReLU hidden layer: std = sqrt(2 / n_in)
    const heStd = Math.sqrt(2 / nFeatures);
    this.W1 = Array.from({ length: nFeatures }, () =>
      Array.from({ length: hiddenSize }, () => gaussianFrom(rand) * heStd)
    );
    this.b1 = new Array(hiddenSize).fill(0);

    // Xavier initialization for the sigmoid output layer: std = sqrt(1 / n_in)
    const xavierStd = Math.sqrt(1 / hiddenSize);
    this.W2 = Array.from({ length: hiddenSize }, () => [gaussianFrom(rand) * xavierStd]);
    this.b2 = [0];
  }

  forward(X: Matrix): { Z1: Matrix; A1: Matrix; Z2: Matrix; A2: Matrix } {
    const Z1 = addRowVec(matmul(X, this.W1), this.b1);
    const A1 = elemwise(Z1, relu);
    const Z2 = addRowVec(matmul(A1, this.W2), this.b2);
    const A2 = elemwise(Z2, sigmoid);
    return { Z1, A1, Z2, A2 };
  }

  private backward(X: Matrix, y: Matrix, Z1: Matrix, A1: Matrix, A2: Matrix) {
    const n = X.length;

    // Output layer: dL/dZ2 = A2 - y (BCE loss + sigmoid output simplification)
    const dZ2 = sub(A2, y);
    const dW2 = elemwise(matmul(transpose(A1), dZ2), (v) => v / n);
    const db2 = colMean(dZ2);

    // Hidden layer, propagated back through W2 then gated by ReLU'
    const dA1 = matmul(dZ2, transpose(this.W2));
    const reluMask = elemwise(Z1, reluDerivative);
    const dZ1 = elemwiseMul(dA1, reluMask);
    const dW1 = elemwise(matmul(transpose(X), dZ1), (v) => v / n);
    const db1 = colMean(dZ1);

    return { dW1, db1, dW2, db2 };
  }

  /** One full-batch gradient-descent step. Returns the BCE loss BEFORE the update. */
  step(samples: Sample[]): number {
    const X: Matrix = samples.map((s) => [s.x[0], s.x[1]]);
    const y: Matrix = samples.map((s) => [s.label]);

    const { Z1, A1, A2 } = this.forward(X);

    let loss = 0;
    for (let i = 0; i < A2.length; i++) {
      const p = Math.min(Math.max(A2[i][0], 1e-9), 1 - 1e-9);
      const t = y[i][0];
      loss += -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
    }
    loss /= A2.length;
    this.lossHistory.push(loss);

    const grads = this.backward(X, y, Z1, A1, A2);
    this.W1 = sub(this.W1, elemwise(grads.dW1, (v) => v * this.lr));
    this.b1 = this.b1.map((v, j) => v - this.lr * grads.db1[j]);
    this.W2 = sub(this.W2, elemwise(grads.dW2, (v) => v * this.lr));
    this.b2 = this.b2.map((v, j) => v - this.lr * grads.db2[j]);

    this.step_ += 1;
    return loss;
  }

  predictProba(x1: number, x2: number): number {
    const { A2 } = this.forward([[x1, x2]]);
    return A2[0][0];
  }

  accuracy(samples: Sample[]): number {
    let correct = 0;
    for (const s of samples) {
      const pred = this.predictProba(s.x[0], s.x[1]) >= 0.5 ? 1 : 0;
      if (pred === s.label) correct += 1;
    }
    return correct / samples.length;
  }
}
