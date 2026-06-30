# Forma

**"Netron lets you see your model. Forma lets you shape it."**

A local-first, open-source GUI for loading trained neural network models, visualizing the computation graph, and applying quantization and pruning operations with live feedback -- no Python scripts, no YAML configs, no CLI flags required.

Repository: https://github.com/Hussain004/Forma

---

## What Forma Does

Every serious model-compression tool today is code-first. ONNX Runtime quantization, Microsoft Olive, NNI, and PyTorch's `torch.ao.quantization` all require scripting. The one widely-used visual tool, Netron, is strictly read-only.

Forma merges inspection with intervention. Load a model, click a layer, quantize it, see the size and accuracy impact immediately, export the result.

**Target users:**
- ML engineers deploying to edge, mobile, or browser targets
- Indie developers shipping local-AI features
- Students learning what quantization does to a network at the layer level

---

## Feature Scope

### v1.0 -- MVP

- Load an `.onnx` model file via drag-and-drop
- Render the computation graph as an interactive, zoomable node diagram
- Per-node inspector: op type, parameter count, estimated size (MB), input/output shapes
- Whole-model dynamic INT8 quantization (one click, no calibration data required)
- Whole-model static INT8 quantization (requires a small calibration dataset: 20 to 100 representative inputs)
- Before/after comparison: file size, parameter count, output similarity score
- Local benchmark: measure inference latency on the user's hardware via ONNX Runtime Web
- Export the modified model as `.onnx`

### v2 -- Differentiated Controls

- Per-layer quantization exclusion (maps to `nodes_to_exclude` in ONNX Runtime)
- Sensitivity analysis: color-code layers green/yellow/red before quantization based on weight distribution heuristics
- Structured pruning: channel/filter pruning with sparsity slider and live size estimate
- Mixed-precision search: given a target size or accuracy floor, suggest per-layer bit-width assignments

### v3 -- Ecosystem

- GGUF export for LLM quantization (GPTQ/AWQ-style)
- Quantization-aware fine-tuning loop
- Plugin system for custom compression passes
- Side-by-side model diff mode

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend framework | React 19 + TypeScript + Vite |
| Graph visualization | React Flow (@xyflow/react) with dagre auto-layout |
| ONNX inference | onnxruntime-web (WebAssembly, runs in a Web Worker) |
| Styling | CSS custom properties, strict 4/8px grid, JetBrains Mono font |
| Testing | Vitest + @testing-library/react |
| Hosting | Vercel (SPA routing via vercel.json rewrite rules) |

The design principle: zero friction. No installing runtimes, no editing config files. Drag a file in, get a usable result in under a minute for the simple case, with progressively deeper controls available for power users.

---

## Architecture

```
Browser
  ModelDropzone (drag .onnx file)
       |
       v
  onnxWorker.ts (Web Worker)
    - loads onnxruntime-web
    - parses graph to OnnxNode[] / OnnxEdge[]
    - runs inference sessions off the main thread
       |
       v
  GraphCanvas.tsx (React Flow)
    - renders nodes and edges
    - dagre layout engine
    - custom OperatorNode and IONode types
       |
  LayerInspector.tsx (right panel)
    - selected node details
    - op type, shapes, param count, estimated size
```

The ONNX Runtime Web worker bridge exposes a typed `useOnnxWorker` React hook. All WASM execution runs off the main thread so the UI stays responsive during model load and inference.

---

## Local Development

### Prerequisites

- Node.js 18 or later
- npm 9 or later

### Install

```bash
git clone https://github.com/Hussain004/Forma.git
cd Forma
npm install
```

### Run dev server

```bash
npm run dev
```

The dev server starts at `http://localhost:5173`. The COOP/COEP headers required for SharedArrayBuffer (WASM threading) are configured in `vite.config.ts`.

### Run tests

```bash
npm test
```

Tests use Vitest with jsdom. The test suite follows TDD: failing tests for graph editing logic and ONNX execution are written before features are implemented.

### Build for production

```bash
npm run build
```

Output goes to `dist/`. The `vercel.json` rewrite rule routes all paths to `index.html` so client-side routing works correctly on Vercel.

---

## Project Structure

```
Forma/
  src/
    components/
      GraphCanvas.tsx       - React Flow graph rendering with dagre layout
      LayerInspector.tsx    - Right-panel node detail view
      ModelDropzone.tsx     - Drag-and-drop .onnx file loader
    hooks/
      useOnnxWorker.ts      - Typed hook wrapping the ONNX Web Worker
    lib/
      onnxParser.ts         - Extracts OnnxNode[]/OnnxEdge[] from model
    workers/
      onnxWorker.ts         - Web Worker: loads onnxruntime-web, handles postMessage
    styles/
      theme.css             - CSS custom properties, Avionics Blueprint design system
    __tests__/
      graph.test.ts         - TDD tests for graph editing logic
      onnx.test.ts          - TDD tests for ONNX execution pipeline
  public/
  vercel.json               - SPA rewrite rules + COOP/COEP headers
  vite.config.ts            - Vite config with worker, WASM, and test settings
  FORMA_IMPLEMENTATION.md   - Full product and architecture specification
```

---

## Design System

The UI follows an Avionics Blueprint theme. Rules that are strictly enforced:

- **Font:** JetBrains Mono, Space Mono, or Fira Code. No Inter, Roboto, or standard sans-serifs.
- **Background:** `#12161A` (deep tactical graphite)
- **Borders:** 1px solid `rgba(255, 255, 255, 0.15)`. No box-shadows. No border-radius above 2px.
- **Active tensor flow:** `#FFB000` (tactical amber)
- **Success state:** `#4A5D23` (military green)
- **Spacing:** All padding and margin values are multiples of 4px. No arbitrary values.
- **No gradients, no glowing borders, no hover lift animations.**

The UI is designed to look like a high-density physical engineering terminal, not a consumer web app.

---

## Deployment

Push to any Vercel-connected branch. The `vercel.json` handles:
1. SPA routing -- all paths rewrite to `/index.html`
2. COOP and COEP security headers -- required for WASM SharedArrayBuffer support

---

## Build Plan

| Phase | Scope |
|---|---|
| 1 | ONNX parsing + static graph visualization |
| 2 | Whole-model dynamic INT8 quantization + size comparison |
| 3 | Static quantization with calibration upload + latency benchmark |
| 4 | Per-layer selection and exclude-from-quantization controls |
| 5 | Pruning support + sensitivity coloring |
| 6 | Packaging, final docs, demo |

---

## Honest Limitations

- The output similarity score on calibration data is a proxy metric, not a real accuracy benchmark. The UI makes this explicit.
- ONNX format only in v1. PyTorch input is treated as "convert to ONNX first" via `torch.onnx.export`, which does not handle all architectures.
- Competing with Netron's format breadth (TensorFlow, Keras, Core ML, Caffe, and more) is not a short-term goal. Forma wins on ONNX depth first.

---

## License

MIT
