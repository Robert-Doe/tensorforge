import './style.css';
import { TwoLayerNet, makeMoons, makeXor, type Sample } from './neuralNet';

const app = document.querySelector<HTMLDivElement>('#app')!;

app.innerHTML = `
  <div class="topbar">
    <div class="brand">⚡ tensorforge</div>
    <div class="links">
      <a href="https://github.com/Robert-Doe/tensorforge" target="_blank" rel="noopener">GitHub</a>
      <a href="https://robertdoe.com">← robertdoe.com</a>
    </div>
  </div>

  <div class="hero">
    <h1>Neural Net <span class="accent">From Scratch</span>, Live</h1>
    <p class="tagline">
      A 2-layer network (input, ReLU hidden layer, sigmoid output) trained by
      backpropagation and gradient descent, ported line-for-line from
      <code>module_17/backprop.py</code>. Every frame here is a real forward pass,
      backward pass, and weight update running in your browser.
    </p>
  </div>

  <div class="section">
    <div class="section-head">
      <div class="eyebrow">Module 17 &middot; Backpropagation From Scratch &middot; Live Port</div>
      <h2>Watch it actually learn</h2>
      <p>
        Pick a non-linearly-separable dataset, then run training. The decision boundary
        reshapes as the hidden layer's weights update, and the loss curve tracks real
        binary cross-entropy going down &mdash; the same math as the Python source
        (He init for the ReLU layer, Xavier init for the sigmoid output, dZ2 = A2 &minus; y).
      </p>
    </div>

    <div class="demo-grid">
      <div class="demo-card">
        <div class="controls">
          <select id="dataset">
            <option value="moons">interleaving moons</option>
            <option value="xor">XOR corners</option>
          </select>
          <button id="btn-run">Train</button>
          <button id="btn-step" class="secondary">Step</button>
          <button id="btn-reset" class="secondary">Reset / Retrain</button>
          <label class="field">
            learning rate
            <input type="range" id="lr" min="0.05" max="2" step="0.05" value="0.5" />
            <span id="lr-val">0.50</span>
          </label>
        </div>

        <div class="canvas-row">
          <div class="canvas-block">
            <h3>Decision boundary</h3>
            <canvas id="canvas-boundary" width="440" height="380"></canvas>
            <div class="legend">
              <span><span class="dot" style="background:#7dd3fc"></span>class 0</span>
              <span><span class="dot" style="background:#d4a017"></span>class 1</span>
            </div>
          </div>
          <div class="canvas-block">
            <h3>Loss curve (BCE)</h3>
            <canvas id="canvas-loss" width="320" height="380"></canvas>
          </div>
        </div>

        <div class="stats">
          <div class="stat"><span class="k">Step</span><span class="v" id="stat-step">0</span></div>
          <div class="stat"><span class="k">Loss</span><span class="v accent" id="stat-loss">–</span></div>
          <div class="stat"><span class="k">Train accuracy</span><span class="v" id="stat-acc">–</span></div>
        </div>

        <div class="arch-note">2 inputs &rarr; 8 hidden (ReLU) &rarr; 1 output (sigmoid) &middot; He/Xavier init &middot; full-batch gradient descent</div>
      </div>
    </div>
  </div>

  <div class="footer">
    <span>Real TypeScript port of <code>module_17/backprop.py</code>'s <code>TwoLayerNet</code> &mdash; forward pass + backprop, no libraries.</span>
    <span><code>tensorforge/webapp</code></span>
  </div>
`;

const canvasB = document.querySelector<HTMLCanvasElement>('#canvas-boundary')!;
const ctxB = canvasB.getContext('2d')!;
const canvasL = document.querySelector<HTMLCanvasElement>('#canvas-loss')!;
const ctxL = canvasL.getContext('2d')!;

const datasetSel = document.querySelector<HTMLSelectElement>('#dataset')!;
const btnRun = document.querySelector<HTMLButtonElement>('#btn-run')!;
const btnStep = document.querySelector<HTMLButtonElement>('#btn-step')!;
const btnReset = document.querySelector<HTMLButtonElement>('#btn-reset')!;
const lrSlider = document.querySelector<HTMLInputElement>('#lr')!;
const lrVal = document.querySelector<HTMLSpanElement>('#lr-val')!;
const statStep = document.querySelector<HTMLSpanElement>('#stat-step')!;
const statLoss = document.querySelector<HTMLSpanElement>('#stat-loss')!;
const statAcc = document.querySelector<HTMLSpanElement>('#stat-acc')!;

const BOUND = 2.2;

function buildDataset(): Sample[] {
  const seed = Math.floor(Math.random() * 1e9);
  return datasetSel.value === 'xor' ? makeXor(160, seed) : makeMoons(160, seed);
}

let data: Sample[] = buildDataset();
let net = new TwoLayerNet(2, 8, parseFloat(lrSlider.value), Math.floor(Math.random() * 1e9));
let running = false;
let rafId: number | null = null;

function toCanvas(cv: HTMLCanvasElement, x: number, y: number): [number, number] {
  const px = ((x + BOUND) / (2 * BOUND)) * cv.width;
  const py = cv.height - ((y + BOUND) / (2 * BOUND)) * cv.height;
  return [px, py];
}

function drawBoundary(): void {
  const w = canvasB.width;
  const h = canvasB.height;
  ctxB.fillStyle = '#141416';
  ctxB.fillRect(0, 0, w, h);

  // probability field
  const cell = 8;
  for (let py = 0; py < h; py += cell) {
    for (let px = 0; px < w; px += cell) {
      const x = (px / w) * (2 * BOUND) - BOUND;
      const y = BOUND - (py / h) * (2 * BOUND);
      const p = net.predictProba(x, y);
      const alpha = Math.abs(p - 0.5) * 0.55;
      ctxB.fillStyle = p >= 0.5 ? `rgba(212,160,23,${alpha})` : `rgba(125,211,252,${alpha})`;
      ctxB.fillRect(px, py, cell, cell);
    }
  }

  // gridlines
  ctxB.strokeStyle = 'rgba(154,153,146,0.15)';
  ctxB.lineWidth = 1;
  for (let g = -Math.floor(BOUND); g <= Math.floor(BOUND); g++) {
    const [gx] = toCanvas(canvasB, g, 0);
    const [, gy] = toCanvas(canvasB, 0, g);
    ctxB.beginPath();
    ctxB.moveTo(gx, 0);
    ctxB.lineTo(gx, h);
    ctxB.moveTo(0, gy);
    ctxB.lineTo(w, gy);
    ctxB.stroke();
  }

  // points
  for (const s of data) {
    const [px, py] = toCanvas(canvasB, s.x[0], s.x[1]);
    ctxB.beginPath();
    ctxB.arc(px, py, 4.5, 0, Math.PI * 2);
    ctxB.fillStyle = s.label === 1 ? '#d4a017' : '#7dd3fc';
    ctxB.fill();
    ctxB.strokeStyle = 'rgba(0,0,0,0.35)';
    ctxB.lineWidth = 1;
    ctxB.stroke();
  }
}

function drawLoss(): void {
  const w = canvasL.width;
  const h = canvasL.height;
  ctxL.fillStyle = '#141416';
  ctxL.fillRect(0, 0, w, h);

  const hist = net.lossHistory;
  ctxL.strokeStyle = 'rgba(154,153,146,0.15)';
  ctxL.lineWidth = 1;
  for (let i = 1; i <= 4; i++) {
    const y = (h / 5) * i;
    ctxL.beginPath();
    ctxL.moveTo(0, y);
    ctxL.lineTo(w, y);
    ctxL.stroke();
  }

  if (hist.length < 2) return;

  const maxLoss = Math.max(...hist, 0.05);
  const windowSize = 400;
  const start = Math.max(0, hist.length - windowSize);
  const visible = hist.slice(start);
  const stepX = w / Math.max(1, windowSize - 1);

  ctxL.strokeStyle = '#d4a017';
  ctxL.lineWidth = 2;
  ctxL.beginPath();
  visible.forEach((loss, i) => {
    const x = i * stepX;
    const y = h - (loss / maxLoss) * (h - 10) - 5;
    if (i === 0) ctxL.moveTo(x, y);
    else ctxL.lineTo(x, y);
  });
  ctxL.stroke();

  ctxL.fillStyle = '#9a9992';
  ctxL.font = '11px JetBrains Mono, monospace';
  ctxL.fillText(`max ${maxLoss.toFixed(3)}`, 6, 14);
}

function render(): void {
  drawBoundary();
  drawLoss();
}

function updateStats(loss: number | null): void {
  statStep.textContent = String(net.step_);
  statLoss.textContent = loss === null ? '–' : loss.toFixed(4);
  statAcc.textContent = `${(net.accuracy(data) * 100).toFixed(1)}%`;
}

function stepAndRender(): void {
  const loss = net.step(data);
  render();
  updateStats(loss);
}

function loop(): void {
  if (!running) return;
  const loss = net.step(data);
  render();
  updateStats(loss);
  if (loss < 0.01 || net.step_ > 4000) {
    stopRun();
    return;
  }
  rafId = requestAnimationFrame(loop);
}

function startRun(): void {
  running = true;
  btnRun.textContent = 'Pause';
  loop();
}

function stopRun(): void {
  running = false;
  btnRun.textContent = 'Train';
  if (rafId !== null) cancelAnimationFrame(rafId);
}

function resetAll(): void {
  stopRun();
  data = buildDataset();
  net = new TwoLayerNet(2, 8, parseFloat(lrSlider.value), Math.floor(Math.random() * 1e9));
  render();
  updateStats(null);
}

btnRun.addEventListener('click', () => {
  if (running) stopRun();
  else startRun();
});

btnStep.addEventListener('click', () => {
  if (running) stopRun();
  stepAndRender();
});

btnReset.addEventListener('click', resetAll);
datasetSel.addEventListener('change', resetAll);

lrSlider.addEventListener('input', () => {
  net.lr = parseFloat(lrSlider.value);
  lrVal.textContent = net.lr.toFixed(2);
});

render();
updateStats(null);
