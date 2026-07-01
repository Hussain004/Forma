<div align="center">

<img src="public/favicon.svg" width="80" alt="Forma" />

# Forma

### Browser-Native ONNX Model Visualizer

**Inspect, analyze, and export neural network models entirely in the browser. No Python. No server. No installation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Version](https://img.shields.io/badge/version-0.5.0-FFB000.svg)](https://github.com/Hussain004/Forma/releases)

[**Live Application**](https://forma-ml.vercel.app) · [Issues](https://github.com/Hussain004/Forma/issues) · [Releases](https://github.com/Hussain004/Forma/releases)

</div>

---

## Overview

Forma is a fully client-side web application for loading, visualizing, and analyzing ONNX neural network models. Drop a `.onnx` file onto the canvas and the complete computation graph renders immediately: nodes laid out automatically with dagre, every tensor edge routed, each operator inspectable with a single click.

All computation runs in the browser via WebAssembly. Models never leave the user's machine.

---

## Capabilities

### Graph Visualization

- Drag-and-drop `.onnx` loading with real-time progress indication
- Automatic top-down layout via dagre; handles arbitrarily deep and wide graphs
- Pan, zoom, and minimap navigation for large models
- Distinct visual treatment for operator nodes versus input/output tensor nodes
- Sensitivity coloring: node border color reflects parameter density (low to critical)
- Filter nodes by operator type or name with live dimming of non-matching nodes
- Jump to the first matching node by pressing Enter in the filter field
- Keyboard shortcuts: `/` focuses the filter input, Escape clears and deselects

### Model Inspection

- Per-node Layer Inspector: operator type, parameter count, estimated weight size in MB, tensor shape annotations for all inputs and outputs
- Op type histogram: model-wide breakdown of every operator category and its frequency, shown when no node is selected
- INT8 size estimate: projected model size after dynamic quantization, displayed in the stats bar and per-node in the inspector
- Inference benchmark: runs a forward pass in the WASM runtime and reports median latency across multiple trials
- Node exclusion: mark individual nodes as excluded from analysis or export

### Export

- Download the original model buffer as exported by the WASM runtime
- Export is performed off-thread; the UI remains responsive throughout

### Engineering

- Off-main-thread ONNX inference via `onnxruntime-web` in a dedicated Web Worker
- Schema-aware binary protobuf parser for full graph metadata extraction
- Typed postMessage protocol between hook and worker with structured error propagation
- `SharedArrayBuffer` multi-threading via COOP/COEP headers
- 104 tests across 6 files; zero TypeScript errors on strict mode

---

## Quick Start

### Use the Live Application

1. Open [**forma-ml.vercel.app**](https://forma-ml.vercel.app)
2. Drag any `.onnx` model file onto the canvas
3. Click any node to inspect it in the Layer Inspector panel

### Run Locally

```bash
git clone https://github.com/Hussain004/Forma.git
cd Forma
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173).

**Requirements:** Node.js 18+. No Python, no CUDA, no native extensions.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | React 19 + TypeScript (strict) + Vite 8 |
| Graph rendering | @xyflow/react (React Flow v12) |
| Layout | dagre |
| ONNX execution | onnxruntime-web (WASM) in a Web Worker |
| Protobuf parsing | Schema-aware binary decoder (no generated code) |
| Design system | CSS custom properties, 4px grid, JetBrains Mono |
| Testing | Vitest + @testing-library/react |
| Deployment | Vercel (static SPA with COOP/COEP headers) |

---

## Architecture

```
Browser (main thread)
|
+-- App.tsx
|     useOnnxWorker hook    (status: idle -> loading -> ready -> benchmarking -> exporting)
|     SelectableGraph state (pure immutable transforms: selectNode, filterGraph, excludeNode)
|     |
|     +-- GraphCanvas       React Flow, dagre layout, OperatorNode + IONode, MiniMap
|     |
|     +-- LayerInspector    Per-node detail, op histogram when no node selected
|     |
|     +-- ModelDropzone     Drag-and-drop with progress bar
|
|                           postMessage (ArrayBuffer transfer, zero-copy)
|
+-- onnxWorker.ts (Web Worker)
      onnxruntime-web WASM
      parseOnnxGraph()  -> OnnxNode[], OnnxEdge[], graphInputs (shapes)
      LOAD_MODEL        -> MODEL_LOADED + QUANTIZE_ESTIMATE
      BENCHMARK         -> BENCHMARK_RESULT
      EXPORT            -> EXPORT_RESULT (ArrayBuffer transfer)
```

**Web Worker isolation:** WASM model loading and inference are blocking operations. Isolating them in a worker keeps the UI at 60 fps regardless of model size. The `useOnnxWorker` hook exposes a clean async interface with typed status transitions.

**No backend:** The entire pipeline runs in the browser. Zero infrastructure, zero server latency, models never leave the user's machine.

**COOP/COEP headers:** `SharedArrayBuffer` requires a cross-origin isolated context. Both `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` are set via `vercel.json` on every response.

---

## Design System

Avionics Blueprint visual language: the aesthetic of a high-density engineering terminal.

| Token | Value | Usage |
|---|---|---|
| Background | `#12161A` | Application base |
| Surface | `#16191C` | Panels, node cards |
| Raised | `#1C2128` | Input/output tensor nodes |
| Amber | `#FFB000` | Active flows, selections, borders |
| Green | `#52C57A` | Success and confirmation states |
| Error | `#C0392B` | Parse failures, load errors |
| Font | JetBrains Mono | All text at all sizes |
| Base unit | 4px | All spacing is a multiple of 4 |
| Border radius | 2px maximum | No rounded cards |

No box-shadows. No gradients. No Inter or Roboto.

---

## Project Structure

```
src/
  components/
    GraphCanvas.tsx       React Flow canvas, dagre layout, MiniMap, JumpController
    LayerInspector.tsx    Per-node detail panel; model summary histogram
    ModelDropzone.tsx     Drag-and-drop with progress indication
  hooks/
    useOnnxWorker.ts      Typed React hook wrapping the ONNX Web Worker
  lib/
    onnxTypes.ts          OnnxNode, OnnxEdge, OnnxGraph interfaces
    onnxProtoParser.ts    Binary protobuf parser for ONNX ModelProto
    graphUtils.ts         Pure graph transforms and utilities
    quantize.ts           INT8 size estimation and formatting
  workers/
    onnxWorker.ts         Web Worker: LOAD_MODEL, BENCHMARK, EXPORT
  __tests__/
    graph.test.ts         Graph utilities and selection model
    onnx.test.ts          Worker lifecycle and message contract
    app.test.tsx          App integration: load flow, selection, error states
    v3.test.ts            Filter, exclusion, INT8 estimation
    v4.test.ts            Export reliability, quantize formatting, download
    v0.5.test.ts          computeOpCounts, keyboard shortcuts, op histogram
```

---

## Development

```bash
npm run dev      # Dev server with COOP/COEP headers
npm test         # 104 tests across 6 files
npx tsc --noEmit # Type-check without building
npm run build    # Production build
```

---

## Releases

| Version | Scope |
|---|---|
| 0.5.0 | MiniMap, jump-to-node, keyboard shortcuts, op type histogram |
| 0.4.0 | INT8 estimate in UI, Download button, export promise hardening |
| 0.3.0 | Graph filter, node exclusion, INT8 size estimate, model export |
| 0.2.1 | Icon update, DEVELOPMENT.md |
| 0.2.0 | Schema-aware protobuf parser, sensitivity coloring, inference benchmark |
| 0.1.0 | MVP: ONNX loading, graph visualization, Layer Inspector |

---

## Limitations

- **ONNX only.** PyTorch `.pt`, `.safetensors`, and other formats are not supported. Convert to ONNX first using `torch.onnx.export`.
- **Graph internals depend on runtime exposure.** `onnxruntime-web` does not expose a public API for reading graph node metadata. Forma uses a schema-aware binary parser as the primary path with a runtime-extraction fallback.
- **INT8 estimates are projections.** The quantization size figures are computed analytically from parameter counts, not from running a quantizer. They are labeled as estimates.

---

## Acknowledgments

- [ONNX Runtime](https://onnxruntime.ai/) for the WebAssembly inference backend
- [React Flow](https://reactflow.dev/) for the interactive graph rendering primitives
- [dagre](https://github.com/dagrejs/dagre) for automatic directed graph layout

---

<div align="center">

Built for ML engineers who need to understand and optimize their models without leaving the browser.

If Forma is useful to you, consider [supporting development](https://donatr.ee/hussain/).

</div>
