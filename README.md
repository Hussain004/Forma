<div align="center">

<img src="public/favicon.svg" width="80" alt="Forma" />

# Forma

### Browser-Native ONNX & TFLite Model Visualizer

**Inspect, analyze, and export neural network models entirely in the browser. No Python. No server. No installation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178c6.svg)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Version](https://img.shields.io/badge/version-1.3.0-FFB000.svg)](https://github.com/Hussain004/Forma/releases)

[**Live Application**](https://forma-ml.vercel.app) · [Issues](https://github.com/Hussain004/Forma/issues) · [Releases](https://github.com/Hussain004/Forma/releases)

</div>

---

## Overview

Forma is a fully client-side web application for loading, visualizing, and analyzing ONNX and TFLite neural network models. Drop a `.onnx` or `.tflite` file onto the canvas and the complete computation graph renders immediately: nodes laid out automatically with dagre, every tensor edge routed, each operator inspectable with a single click. ONNX models are fully editable and exportable; TFLite support is read-only.

All computation runs in the browser via WebAssembly. Models never leave the user's machine.

---

## Capabilities

### Graph Visualization

- Drag-and-drop `.onnx` or `.tflite` loading with real-time progress indication
- Automatic layout via dagre with TB (top-down) and LR (left-right) toggle
- Pan, zoom, and minimap navigation for large models
- Distinct visual treatment for operator nodes versus input/output tensor nodes
- Op category coloring: each node's left accent bar indicates its operator category (Convolution, Activation, Normalization, Linear, Pooling, Reshape, and more)
- Search dropdown: live-filtered results with keyboard navigation (arrow keys, Enter to jump, Escape to dismiss)
- Filter nodes by operator type, name, or tensor name with live dimming of non-matching nodes
- Keyboard shortcuts: `/` focuses the filter input, Escape clears and deselects
- Hover tooltip: instant op type, parameter count, and output shape on mouse-over without clicking
- Edge shape labels: selecting a node shows tensor shapes on all edges directly connected to it

### Model Inspection

- Click any node to open the Layer Inspector with operator type, node name, parameter count, estimated weight size, tensor shape annotations, and full attribute listing (kernel size, strides, epsilon, group, auto_pad, and every other op attribute stored in the model)
- Inline attribute editing: click any attribute value to edit it directly; integer, float, string, and array attributes are all editable with type-aware parsing
- Structural editing: delete a node with automatic reconnection, or a picker to choose the reconnection source when it has multiple inputs; insert a passthrough Identity node by clicking any edge. A green "NEW" badge marks inserted nodes in the canvas
- Ctrl+Z undo: step back through attribute and structural edits alike, in the order they were made
- Modified badge: edited nodes are marked with a "MOD" indicator in the canvas and a "Modified" label in the inspector
- Ctrl/Meta+click for multi-select: build a selection across multiple nodes simultaneously
- Aggregate inspector: combined parameter count, total size, and op type breakdown when multiple nodes are selected
- Bulk exclude/include: EXCLUDE ALL and INCLUDE ALL buttons apply to the full selection at once
- Ancestor/descendant trace: selecting a node highlights all upstream producers (blue accent) and downstream consumers (green accent), dimming unrelated nodes
- Op type histogram with graph depth: model-wide operator breakdown sorted by frequency, plus longest-path depth, shown when no node is selected
- Model metadata panel: producer name and version, opset version, and IR version shown in the summary view
- Category legend in model summary showing only operator categories present in the loaded model
- INT8 size estimate: projected model size after dynamic quantization, in the stats bar and per-node in the inspector
- Inference benchmark: forward pass in the WASM runtime with median latency across multiple trials
- Node exclusion: mark individual nodes as excluded; visual strikethrough applied to excluded cards

### TFLite Support (Read-Only)

- Format detected by the file's own identifier bytes, not just its extension, so drag-and-drop works correctly regardless of how the file is named
- Full graph visualization, category-colored nodes, tensor shapes, and weight sizes in the same canvas and Layer Inspector as ONNX -- no separate UI
- Read-only by design: no attribute editing, no node deletion or insertion, no Benchmark or Export Modified. A dim "TFLite read-only" badge in the stats bar makes this explicit; plain Download still works

### Export


- Download the original model buffer as exported by the WASM runtime
- Export Modified: write attribute edits and structural edits (deleted or inserted nodes) back into a valid ONNX binary protobuf and download the patched model
- Initializer weight bytes are preserved byte-for-byte on export; only what changed is re-encoded, everything else passes through untouched
- Inserted nodes are placed to preserve ONNX's required topological node order, so exported files pass strict validation, not just onnxruntime's own lenient loading
- Exported filename strips the original extension cleanly (e.g. `model_export.onnx`, never `model.onnx_export.onnx`)
- Export is performed off-thread; the UI remains responsive throughout
- Copy node metadata to clipboard with a single button press in the Layer Inspector

### Engineering

- Off-main-thread ONNX inference via `onnxruntime-web` in a dedicated Web Worker
- Schema-aware binary protobuf parser for full graph metadata extraction
- Hand-written binary FlatBuffers parser for TFLite, independent of the protobuf parser -- a completely different wire format (table/vtable/offset-based rather than tag/varint-based), verified against the authoritative TFLite schema
- Byte-preserving protobuf writer: patches only the fields that changed, leaving everything else (including large initializer tensors) untouched; structural edits (node delete/insert) use an array-based rewrite that preserves topological node order
- Both parsers build the same graph representation through a shared generic layer, so the graph canvas and inspector need no format-specific code
- Typed postMessage protocol between hook and worker with structured error propagation
- `SharedArrayBuffer` multi-threading via COOP/COEP headers
- 224 tests across 15 files; zero TypeScript errors on strict mode

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
|     +-- GraphCanvas       React Flow, dagre layout, OperatorNode + IONode, MiniMap, hover tooltip
|     |
|     +-- LayerInspector    Per-node detail, multi-select aggregate, model summary histogram
|     |
|     +-- ModelDropzone     Drag-and-drop with progress bar
|
|                           postMessage (ArrayBuffer transfer, zero-copy)
|
+-- onnxWorker.ts (Web Worker)
      onnxruntime-web WASM (ONNX only)
      isTfliteBuffer()  -> format sniff, decides which parser + whether to create a session
      parseOnnxGraph() / parseTfliteGraph()  -> OnnxNode[], OnnxEdge[], graphInputs (shapes)
      LOAD_MODEL        -> MODEL_LOADED + QUANTIZE_ESTIMATE
      BENCHMARK         -> BENCHMARK_RESULT (ONNX only, no TFLite runtime exists)
      EXPORT            -> EXPORT_RESULT (ArrayBuffer transfer)
      EXPORT_MODIFIED   -> EXPORT_RESULT (attribute and structural edits patched into the original buffer, ONNX only)
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
    GraphCanvas.tsx       React Flow canvas, dagre layout, MiniMap, JumpController, hover tooltip
    LayerInspector.tsx    Per-node detail, aggregate multi-select view, model summary histogram
    ModelDropzone.tsx     Drag-and-drop with progress indication
  hooks/
    useOnnxWorker.ts      Typed React hook wrapping the ONNX Web Worker
  lib/
    onnxTypes.ts          OnnxNode, OnnxEdge, OnnxGraph (format: 'onnx' | 'tflite') interfaces
    onnxProtoParser.ts    Binary protobuf parser for ONNX ModelProto
    onnxProtoWriter.ts    Byte-preserving protobuf writer: attribute edits, node delete/insert
    tfliteParser.ts       Binary FlatBuffers parser for TFLite (read-only): FlatBufferReader,
                          BuiltinOperator name table, tensor-index-to-name translation
    onnxParser.ts         buildGraphFromParsed() -- generic ParsedGraph -> OnnxGraph builder
                          shared by both the ONNX and TFLite parsers
    attrUtils.ts          inferAttrType, parseAttrEdit -- attribute type inference and parsing
    graphUtils.ts         Pure graph transforms: selection, filter, exclusion, tracing, depth,
                          delete eligibility, delete-with-reconnect, passthrough insertion,
                          OP_CATEGORIES (ONNX + TFLite op names)
    quantize.ts           INT8 size estimation and formatting
  workers/
    onnxWorker.ts         Web Worker: LOAD_MODEL (format-sniffed), BENCHMARK, EXPORT, EXPORT_MODIFIED
  __tests__/
    graph.test.ts         Graph utilities and selection model
    onnx.test.ts          Worker lifecycle and message contract
    app.test.tsx          App integration: load flow, selection, error states
    v3.test.ts            Filter, exclusion, INT8 estimation
    v4.test.ts            Export reliability, quantize formatting, download
    v0.5.test.ts          computeOpCounts, keyboard shortcuts, op histogram
    v0.6.test.ts          opCategoryColor, getAncestors/getDescendants, computeGraphDepth
    v0.7.test.ts          setMultiSelection, bulkExclude/bulkInclude, aggregate inspector
    v0.8.test.ts          layout toggle, search dropdown, clipboard copy, benchmark types
    v0.9.test.ts          attribute viewer, tensor name search, edge shape labels
    v0.10.test.ts         model metadata, node name, producer/opset/IR version parsing
    v1.0.test.ts          attribute type inference, value parsing, inline editing, MOD badge
    v1.1.test.ts          protobuf writer: int/float/string/array attribute edits, byte preservation
    v1.2.test.ts          structural editing: delete/insert eligibility, reconnection, topological order
    v1.3.test.ts          TFLite: format detection, FlatBuffers fixture round-trip, opcode fallback
```

---

## Development

```bash
npm run dev      # Dev server with COOP/COEP headers
npm test         # 224 tests across 15 files
npx tsc --noEmit # Type-check without building
npm run build    # Production build
```

---

## Releases

| Version | Scope |
|---|---|
| 1.3.0 | TFLite support (read-only): binary FlatBuffers parser, shared graph/canvas/inspector with ONNX |
| 1.2.0 | Structural editing: delete a node with reconnection, insert a passthrough node, both exportable |
| 1.1.0 | Protobuf writer, Export Modified button, byte-preserving attribute patching |
| 1.0.0 | Inline attribute editing, Ctrl+Z undo, MOD badge on edited nodes |
| 0.10.0 | Model metadata (producer, opset, IR version), node names, 3-color favicon |
| 0.9.0 | Attribute viewer, tensor name search, edge shape labels, intermediate tensor shapes |
| 0.8.0 | Layout toggle (TB/LR), search dropdown, clipboard copy, benchmark type fix |
| 0.7.0 | Multi-select, aggregate inspector, bulk exclude/include, hover tooltip |
| 0.6.0 | Op category coloring, ancestor/descendant trace, graph depth stat |
| 0.5.1 | Stacked layers favicon, README rewrite |
| 0.5.0 | MiniMap, jump-to-node, keyboard shortcuts, op type histogram |
| 0.4.0 | INT8 estimate in UI, Download button, export promise hardening |
| 0.3.0 | Graph filter, node exclusion, INT8 size estimate, model export |
| 0.2.1 | Icon update, session guide |
| 0.2.0 | Schema-aware protobuf parser, sensitivity coloring, inference benchmark |
| 0.1.0 | MVP: ONNX loading, graph visualization, Layer Inspector |

---

## Limitations

- **ONNX and TFLite only.** PyTorch `.pt`, `.safetensors`, and other formats are not supported. Convert to ONNX first using `torch.onnx.export` for full editing support.
- **TFLite is read-only.** No attribute editing, structural editing, benchmarking, or Export Modified -- visualization and inspection only. Editing TFLite models is not planned; ONNX remains the primary edit-and-export format.
- **TFLite per-op attributes are not shown.** Decoding them requires walking ~100 distinct per-operator-type FlatBuffers schemas (Conv2DOptions, Pool2DOptions, etc.), which is out of scope for the initial read-only viewer. Topology, tensor shapes, and weight sizes are all shown; op-specific parameters (e.g. Conv2D stride) are not yet.
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
