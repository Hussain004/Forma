# Forma

> Netron lets you see your model. Forma lets you shape it.

Forma is a browser-native, open-source tool for loading trained neural network models, visualizing their computation graph interactively, and applying compression operations -- quantization, pruning, sensitivity analysis -- with immediate, per-layer feedback. No Python environment, no CLI, no configuration files required.

**Live:** [forma.vercel.app](https://forma.vercel.app) &nbsp;|&nbsp; **Source:** [github.com/Hussain004/Forma](https://github.com/Hussain004/Forma)

---

## The Problem

Every serious model-compression tool is code-first:

| Tool | What it does | Interface |
|:--|:--|:--|
| ONNX Runtime quantization | INT8 dynamic / static PTQ | Python API |
| Microsoft Olive | Multi-pass optimization pipelines | JSON config + CLI |
| Microsoft NNI | Pruning, quantization, NAS | Python scripts |
| NVIDIA Model Optimizer | Quantization, sparsity, distillation | Python SDK |
| `torch.ao.quantization` | PTQ / QAT | Python API |

The one popular visual tool, **Netron**, is read-only. It shows you the graph; you cannot act on it.

Forma closes that gap: inspect a model, select a layer, quantize it, see the size and accuracy delta immediately, and export the result. All of this happens in the browser, using WebAssembly, with no server and no installation.

---

## Key Features

**Currently available (v1)**

- Drag-and-drop `.onnx` model loading
- Interactive, zoomable computation graph with automatic layout (dagre, top-down)
- Per-node inspector: operator type, parameter count, estimated weight size, input/output tensor shapes
- Off-main-thread ONNX inference via `onnxruntime-web` in a dedicated Web Worker
- Real-time load progress with stage labels (Loading model, Parsing graph, Ready)

**Planned (v2)**

- Whole-model dynamic INT8 quantization -- one click, no calibration data
- Whole-model static INT8 quantization -- with calibration dataset upload
- Before/after comparison panel: file size delta, parameter count, output similarity score
- Per-layer quantization exclusion via node selection
- Sensitivity coloring: green / yellow / red heat map before you quantize
- Local latency benchmark running both model versions on your hardware

**Planned (v3)**

- Structured pruning with sparsity slider and live size estimate
- Mixed-precision search: auto-suggest per-layer bit-widths given a size or accuracy target
- GGUF export path for LLM quantization
- Side-by-side model diff mode

---

## Quick Start

```bash
git clone https://github.com/Hussain004/Forma.git
cd Forma
npm install
npm run dev
```

Open `http://localhost:5173`, drag any `.onnx` file onto the canvas, and the graph renders immediately.

**Requirements:** Node.js 18+, npm 9+. No Python, no CUDA, no native extensions.

---

## Tech Stack

| Concern | Technology |
|:--|:--|
| Framework | React 19 + TypeScript + Vite 8 |
| Graph layout | @xyflow/react (React Flow) + dagre |
| ONNX execution | onnxruntime-web (WASM, Web Worker) |
| Design system | CSS custom properties, 4/8px grid, JetBrains Mono |
| Testing | Vitest + @testing-library/react (46 tests) |
| Deployment | Vercel -- SPA routing + COOP/COEP headers via `vercel.json` |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Browser (main thread)                              │
│                                                     │
│   ModelDropzone          App.tsx                    │
│   (drag / click)   -->   useOnnxWorker hook         │
│                          SelectableGraph state       │
│                          handleNodeSelect            │
│                                    |                │
│            ┌───────────────────────┤                │
│            |                       |                │
│     GraphCanvas              LayerInspector         │
│     (React Flow +            (selected node         │
│      dagre layout)            detail panel)         │
└────────────────────────┬────────────────────────────┘
                         | postMessage (structured clone / transfer)
┌────────────────────────▼────────────────────────────┐
│  onnxWorker.ts (Web Worker)                         │
│                                                     │
│   onnxruntime-web (WASM)                            │
│   InferenceSession.create(ArrayBuffer)              │
│   parseOnnxGraph() --> OnnxNode[] / OnnxEdge[]      │
│   session.run()    --> Float32Array outputs         │
└─────────────────────────────────────────────────────┘
```

**Why a Web Worker?** ONNX model loading and inference are blocking WASM operations. Isolating them in a worker keeps the UI at 60 fps regardless of model size. The `useOnnxWorker` hook exposes a clean async interface: `loadModel(buffer, filename)` and `runInference(inputs, shapes)`, with typed status transitions (`idle -> loading -> ready -> running`).

**Why no backend?** Forma is intentionally serverless. The entire compression pipeline runs in the browser via `onnxruntime-web`. This means zero infrastructure, zero latency to a server, and models never leave the user's machine.

---

## Design System

Forma uses an **Avionics Blueprint** visual language -- the aesthetic of a high-density physical engineering terminal, not a consumer web application.

| Token | Value | Usage |
|:--|:--|:--|
| Background | `#12161A` | Application base |
| Surface | `#16191C` | Panels, node cards |
| Raised | `#1C2128` | I/O tensor nodes |
| Amber | `#FFB000` | Active tensor flows, selections, active borders |
| Military green | `#4A5D23` | Success / confirmation states |
| Error | `#C0392B` | Parse failures, load errors |
| Text primary | `#E8EAF0` | Labels, values |
| Text secondary | `#8A8F9E` | Metadata, placeholders |
| Font | JetBrains Mono | All text, all sizes |
| Base unit | 4px | All spacing is a multiple of 4 |
| Border | 1px solid rgba(255,255,255,0.15) | All borders, no glow |
| Border radius | 2px max | No rounded cards |

Rules that are never broken: no box-shadows, no gradients, no border-radius above 2px, no Inter or Roboto, no hover-lift animations.

---

## Project Structure

```
Forma/
  src/
    components/
      GraphCanvas.tsx        React Flow canvas, dagre layout, custom node types
      LayerInspector.tsx     Right-panel detail view for a selected node
      ModelDropzone.tsx      Full-viewport drag-and-drop with crosshair idle state
    hooks/
      useOnnxWorker.ts       Typed React hook wrapping the ONNX Web Worker
    lib/
      onnxTypes.ts           Shared OnnxNode / OnnxEdge / OnnxGraph interfaces
      onnxParser.ts          Extracts graph structure from an InferenceSession
      graphUtils.ts          Pure selection helpers: selectNode, deselectAll, validateEdges
    workers/
      onnxWorker.ts          Web Worker: LOAD_MODEL, RUN_INFERENCE, PROGRESS, ERROR
    styles/
      theme.css              CSS custom properties -- all design tokens
    __tests__/
      graph.test.ts          23 unit tests: selection model, edge validation, param sums
      onnx.test.ts           15 unit tests: worker lifecycle, message contract, parser
      app.test.tsx           8 integration tests: load flow, single-select, error states
  vercel.json                SPA rewrite + COOP / COEP headers for SharedArrayBuffer
  vite.config.ts             Worker ES format, onnxruntime-web exclusion, Vitest config
  FORMA_IMPLEMENTATION.md    Full product specification and phased build plan
```

---

## Development

```bash
# Start dev server (with COOP/COEP headers for WASM SharedArrayBuffer)
npm run dev

# Run the full test suite (46 tests across 3 files)
npm test

# Type-check without building
npx tsc -p tsconfig.app.json --noEmit

# Production build
npm run build
```

Tests follow a strict TDD discipline. The graph utility tests and ONNX pipeline tests were written before the implementations existed. The App integration tests run against a mocked Web Worker and verify the full UI state machine from model load through node selection.

---

## Deployment

The project deploys to Vercel with no additional configuration beyond connecting the repository. The `vercel.json` in the repository root handles two requirements automatically:

**SPA routing** -- all URL paths rewrite to `/index.html` so client-side navigation works on hard refresh.

**Security headers** -- `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` are set on every response. These are required by browsers before they enable `SharedArrayBuffer`, which `onnxruntime-web` uses for WASM threading. Without them, the ONNX worker will silently fall back to single-threaded execution.

---

## Roadmap

| Phase | Status | Scope |
|:--|:--|:--|
| 1 | Complete | ONNX model loading, graph visualization, node inspector |
| 2 | Planned | Dynamic + static INT8 quantization, before/after comparison panel |
| 3 | Planned | Per-layer selection, exclude-from-quantization controls |
| 4 | Planned | Sensitivity coloring, local latency benchmark |
| 5 | Planned | Structured pruning, mixed-precision search |
| 6 | Planned | Packaging, demo recording, public launch |

---

## Limitations

- **Accuracy proxy, not ground truth.** The output similarity score (cosine distance on calibration data) is a fast proxy for quantization quality. It is not a substitute for evaluating on a real labeled validation set. Forma labels it clearly as an estimate.
- **ONNX format only in v1.** PyTorch `.pt` and `.safetensors` files are not accepted directly. Convert to ONNX first using `torch.onnx.export`, then load the resulting `.onnx` file. Not all architectures survive that conversion cleanly.
- **Graph internals depend on runtime exposure.** `onnxruntime-web` does not expose a stable public API for reading graph node metadata. Forma accesses the WASM handler internals with a documented fallback path for when those internals are unavailable.
- **No format breadth.** Competing with Netron's support for TensorFlow, Keras, Core ML, Caffe, and a dozen other formats is not a near-term goal. Forma goes deep on ONNX first.

---

## License

MIT
