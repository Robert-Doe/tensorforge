# Tensorforge — web demo

A real, interactive, in-browser port of `module_17/backprop.py`'s `TwoLayerNet`: a
2-input &rarr; 8-hidden(ReLU) &rarr; 1-output(sigmoid) neural network trained with
actual backpropagation and full-batch gradient descent. He initialization for the
ReLU layer, Xavier initialization for the sigmoid output layer, and the standard
`dZ2 = A2 - y` simplification for binary cross-entropy + sigmoid output — all ported
directly from the Python source, running entirely client-side in TypeScript.

Pick a synthetic, non-linearly-separable 2D dataset (interleaving moons or XOR
corners), hit **Train**, and watch the decision boundary reshape in real time while
the loss curve tracks real BCE loss decreasing. **Step** advances one gradient
update at a time; **Reset / Retrain** reinitializes weights and resamples data; the
learning-rate slider changes the step size live.

## Local development

```bash
cd webapp
npm install
npm run dev
```

## Build

```bash
cd webapp
npm install
npm run build
```

Outputs a static site to `webapp/dist/`. Must complete with zero errors before deploying.

## Deploy (static hosting)

- **Root directory:** `webapp`
- **Build command:** `npm run build`
- **Output directory:** `dist`

### Vercel
```bash
vercel --cwd webapp
```
Or via dashboard: import repo, root directory `webapp`, build command `npm run build`, output directory `dist`.

### Netlify
Site settings → Build & deploy:
- Base directory: `webapp`
- Build command: `npm run build`
- Publish directory: `webapp/dist`

### Cloudflare Pages
- Root directory: `webapp`
- Build command: `npm run build`
- Build output directory: `dist`

## Source layout

- `src/neuralNet.ts` — the ported algorithm (forward pass, backprop, He/Xavier init, hand-rolled matrix ops, dataset generators)
- `src/main.ts` — UI wiring, decision-boundary canvas, loss-curve canvas
- `src/style.css` — design system (dark theme, CSS custom properties)
