<div align="center">

<img src="public/favicon.svg" width="80" alt="Forma Logo" />

# Forma

### Browser-Native ONNX Network Visualizer

**Load trained neural networks in your browser. Inspect every layer. No Python, no CLI, no installation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646cff.svg)](https://vite.dev/)
[![Version](https://img.shields.io/badge/version-1.0.0-3b82f6.svg)](https://github.com/Hussain004/Forma/releases)

[**→ Live Demo**](https://forma-ml.vercel.app) · [Report Bug](https://github.com/Hussain004/Forma/issues) · [Request Feature](https://github.com/Hussain004/Forma/issues)

</div>

---

## What is Forma?

**Forma** is a fully static web application for loading trained neural network models, visualizing their computation graph interactively, and applying compression operations with immediate, per-layer feedback. Drag a `.onnx` file onto the canvas and the entire graph renders in the browser: nodes positioned automatically, every tensor edge labeled, each layer inspectable with a single click. No server required, no installation, no account.

The one popular visual tool, Netron, is read-only. It shows you the graph; you cannot act on it. Forma closes that gap: inspect a model, select a layer, quantize it, see the size and accuracy delta immediately, and export the result. All of this happens in the browser using WebAssembly.

### Why Forma?

| Problem | Forma Solution |
|---|---|
| Need a Python environment and CUDA to inspect a model | Works in any modern browser with zero install |
| Netron is read-only, no compression controls | Interactive graph with planned per-layer quantization |
| ONNX Runtime quantization requires a Python API | One-click dynamic INT8 from the UI (v2) |
| Microsoft Olive needs JSON config files and a CLI | Visual, per-layer compression controls (v2 and v3) |
| Can't easily share model structure with a teammate | Zero-install, send anyone the URL |
| Model debugging requires reading raw ONNX protobuf | Click any node for full metadata: op type, parameter count, tensor shapes |
| Students struggle to set up ML tooling | Drag and drop a `.onnx` file and the graph is there |

---

## Features

A condensed feature list is below. Detailed release notes will live in [CHANGELOG.md](CHANGELOG.md) as the project matures.

### Graph visualization

- Drag-and-drop `.onnx` model loading directly in the browser
- Automatic top-down layout via dagre: nodes positioned, edges routed, no manual arrangement
- Pan and zoom across arbitrarily large graphs with no performance degradation
- Distinct visual treatment for operator nodes and input/output tensor nodes
- Amber-colored tensor flow edges on an engineering-terminal dot-grid background
- Edge validation guard prevents dangling edges from reaching the layout engine

### Model inspection

- Click any node to open the Layer Inspector panel
- Per-node detail: operator type, parameter count, estimated weight size in MB, input and output tensor names
- Single-select model: selecting one node deselects all others, pure and testable
- Null-state placeholder when no node is selected, amber active border when a node is selected

### WASM inference pipeline

- Off-main-thread ONNX inference via `onnxruntime-web` in a dedicated Web Worker
- The main thread stays at 60 fps during model loading and inference regardless of model size
- Real-time load progress with stage labels: Loading model, Parsing graph, Ready
- `SharedArrayBuffer` multi-threading enabled via `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers
- Typed postMessage protocol between the hook and the worker: `LOAD_MODEL`, `RUN_INFERENCE`, `PROGRESS`, `ERROR`
- ArrayBuffer transfer (zero-copy) from the main thread to the worker on model load

### Planned compression tools (v2)

- Whole-model dynamic INT8 quantization: one click, no calibration data required
- Whole-model static INT8 quantization: with calibration dataset upload
- Before/after comparison panel: file size delta, parameter count, output similarity score
- Per-layer quantization exclusion via node selection in the graph
- Sensitivity coloring: green/yellow/red heat map showing compression risk before you quantize
- Local latency benchmark running both model versions on your device

### Planned advanced tools (v3)

- Structured pruning with a sparsity slider and live size estimate
- Mixed-precision search: auto-suggest per-layer bit-widths given a size or accuracy target
- GGUF export path for LLM quantization workflows
- Side-by-side model diff mode

### Roadmap

| Phase | Status | Scope |
|---|---|---|
| 1 | Complete | ONNX model loading, graph visualization, per-node layer inspector |
| 2 | Planned | Dynamic and static INT8 quantization, before/after comparison panel |
| 3 | Planned | Per-layer selection and exclusion from quantization |
| 4 | Planned | Sensitivity coloring, local latency benchmark |
| 5 | Planned | Structured pruning, mixed-precision search |
| 6 | Planned | Packaging, demo recording, public launch |

---

## Quick Start

### Use the Live Demo

1. Open [**forma-ml.vercel.app**](https://forma-ml.vercel.app)
2. Drag any `.onnx` model file onto the canvas
3. Click any graph node to inspect it in the Layer Inspector panel

### Run Locally

```bash
git clone https://github.com/Hussain004/Forma.git
cd Forma
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

**Requirements:** Node.js 18+, npm 9+. No Python, no CUDA, no native extensions.

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Framework** | React 19 + TypeScript + Vite 8 | Component-based SPA with fast HMR |
| **Graph rendering** | @xyflow/react (React Flow) | Interactive, pannable, zoomable computation graph |
| **Layout** | dagre | Automatic top-down directed graph layout |
| **ONNX execution** | onnxruntime-web (WASM) | Browser-side model loading and inference |
| **Threading** | Web Worker + SharedArrayBuffer | Off-main-thread WASM with multi-threading support |
| **Design system** | CSS custom properties | 4/8px grid, JetBrains Mono, Avionics Blueprint tokens |
| **Testing** | Vitest + @testing-library/react | 46 tests across 3 files (unit and integration) |
| **CI** | None yet | Planned: GitHub Actions on every PR |
| **Deployment** | Vercel | Static SPA hosting with COOP/COEP headers via `vercel.json` |

---

## Architecture

```
Browser (main thread)
|
|-- ModelDropzone (drag + click)
|       |
|       v
|-- App.tsx
|     useOnnxWorker hook         (status: idle -> loading -> ready -> running)
|     SelectableGraph state      (pure helpers: selectNode, deselectAll, validateEdges)
|     handleNodeSelect
|     |
|     |---- GraphCanvas          React Flow + dagre layout, custom OperatorNode and IONode types
|     |
|     `---- LayerInspector       Selected node detail panel (op type, params, size, tensors)
|
|                                postMessage (structured clone + ArrayBuffer transfer)
|
`-- onnxWorker.ts (Web Worker)
      onnxruntime-web (WASM)
      InferenceSession.create(ArrayBuffer)
      parseOnnxGraph()  -> OnnxNode[] + OnnxEdge[]
      session.run()     -> Float32Array outputs
```

**Why a Web Worker?** ONNX model loading and inference are blocking WASM operations. Isolating them in a dedicated worker keeps the UI at 60 fps regardless of model size. The `useOnnxWorker` hook exposes a clean async interface: `loadModel(buffer, filename)` and `runInference(inputs, shapes)`, with typed status transitions (`idle -> loading -> ready -> running`).

**Why no backend?** Forma is intentionally serverless. The entire pipeline runs in the browser via `onnxruntime-web`. Zero infrastructure, zero server latency, and models never leave the user's machine.

**Why COOP/COEP headers?** `SharedArrayBuffer` is only available in cross-origin isolated contexts. Both `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` are required before browsers enable it. `onnxruntime-web` uses `SharedArrayBuffer` for its WASM threading backend. The `vercel.json` sets both headers on every response so this works transparently in production and in local dev.

---

## Design System

Forma uses an Avionics Blueprint visual language: the aesthetic of a high-density physical engineering terminal, not a consumer web application.

| Token | Value | Usage |
|---|---|---|
| Background | `#12161A` | Application base |
| Surface | `#16191C` | Panels, node cards |
| Raised | `#1C2128` | Input/output tensor nodes |
| Amber | `#FFB000` | Active tensor flows, selections, active borders |
| Military green | `#4A5D23` | Success and confirmation states |
| Error | `#C0392B` | Parse failures, load errors |
| Text primary | `#E8EAF0` | Labels, values |
| Text secondary | `#8A8F9E` | Metadata, placeholders |
| Font | JetBrains Mono | All text, all sizes |
| Base unit | 4px | All spacing is a multiple of 4 |
| Border | 1px solid rgba(255,255,255,0.15) | All borders, no glow |
| Border radius | 2px maximum | No rounded cards |

Rules that are never broken: no box-shadows, no gradients, no border-radius above 2px, no Inter or Roboto, no hover-lift animations.

---

## Project Structure

```
Forma/
  src/
    components/
      GraphCanvas.tsx        React Flow canvas, dagre TB layout, custom OperatorNode and IONode types
      LayerInspector.tsx     Right-panel detail view for a selected node
      ModelDropzone.tsx      Full-viewport drag-and-drop with SVG crosshair idle state
    hooks/
      useOnnxWorker.ts       Typed React hook wrapping the ONNX Web Worker
    lib/
      onnxTypes.ts           Shared OnnxNode / OnnxEdge / OnnxGraph interfaces
      onnxParser.ts          Extracts graph structure from an InferenceSession
      graphUtils.ts          Pure helpers: toSelectableGraph, selectNode, deselectAll, validateEdges
    workers/
      onnxWorker.ts          Web Worker: handles LOAD_MODEL, RUN_INFERENCE, PROGRESS, ERROR
    styles/
      theme.css              CSS custom properties, all Avionics Blueprint design tokens
    __tests__/
      graph.test.ts          23 unit tests: selection model, edge validation, parameter sums
      onnx.test.ts           15 unit tests: worker lifecycle, message contract, parser fallback
      app.test.tsx           8 integration tests: load flow, single-select, error states
  vercel.json                SPA rewrite rules and COOP/COEP headers for SharedArrayBuffer
  vite.config.ts             Worker ES format, onnxruntime-web exclusion, Vitest configuration
  FORMA_IMPLEMENTATION.md    Full product specification and phased build plan
```

---

## Development

```bash
# Start dev server (includes COOP/COEP headers for WASM SharedArrayBuffer)
npm run dev

# Run the full test suite (46 tests across 3 files)
npm test

# Type-check without building
npx tsc -p tsconfig.app.json --noEmit

# Production build
npm run build
```

Tests follow strict TDD discipline. The graph utility tests and ONNX pipeline tests were written before the implementations existed. The App integration tests run against a mocked Web Worker and verify the full UI state machine from model load through node selection.

---

## Deployment

The project deploys to Vercel with no additional configuration beyond connecting the repository. The `vercel.json` in the repository root handles two requirements automatically.

**SPA routing:** all URL paths rewrite to `/index.html` so client-side navigation works on hard refresh.

**Security headers:** `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` are set on every response. These are required by browsers before they enable `SharedArrayBuffer`, which `onnxruntime-web` uses for WASM threading. Without them, the ONNX worker silently falls back to single-threaded execution.

---

## Limitations

- **ONNX format only in v1.** PyTorch `.pt` and `.safetensors` files are not accepted directly. Convert to ONNX first using `torch.onnx.export`, then load the resulting `.onnx` file. Not all architectures survive that conversion cleanly.
- **Graph internals depend on runtime exposure.** `onnxruntime-web` does not expose a stable public API for reading graph node metadata. Forma accesses WASM handler internals with a documented fallback path for when those internals are unavailable.
- **Accuracy proxy, not ground truth.** The output similarity score planned for v2 is a fast proxy for quantization quality. It is not a substitute for evaluating on a real labeled validation set. Forma will label it clearly as an estimate.
- **No format breadth.** Competing with Netron's support for TensorFlow, Keras, Core ML, Caffe, and a dozen other formats is not a near-term goal. Forma goes deep on ONNX first.

---

## Acknowledgments

- [ONNX Runtime](https://onnxruntime.ai/) for the WebAssembly inference backend that makes browser-side model execution possible
- [React Flow](https://reactflow.dev/) for the interactive graph rendering primitives
- [dagre](https://github.com/dagrejs/dagre) for automatic directed graph layout

---

<div align="center">

Built for ML engineers who want to understand and optimize their models without leaving the browser.

If Forma saves you time, consider giving it a star on [GitHub](https://github.com/Hussain004/Forma).

</div>
