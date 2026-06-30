# Forma: Development Status

This file is the canonical record of what has been built, how it works, and what comes next.
New sessions should read this before touching any code.

---

## Current state: v2 merged to master

Vercel deployment: https://forma-ml.vercel.app
GitHub: https://github.com/Hussain004/Forma
Branch strategy: feature work on versioned branches (v2, v3, ...), PR to master when done.

---

## Architecture overview

Forma is a fully browser-native SPA. There is no backend. Every operation runs in the browser.

```
App.tsx  (React, main thread)
  |-- ModelDropzone      drag-and-drop, FileReader -> ArrayBuffer
  |-- StatsBar           model name / params / size / node count / benchmark result
  |-- GraphCanvas        React Flow + dagre layout, OperatorNode + IONode
  |-- LayerInspector     selected node detail panel
  |
  useOnnxWorker hook     manages the Web Worker lifecycle
  |
  onnxWorker.ts (Web Worker, off-main-thread)
    |-- onnxProtoParser  binary protobuf parse of the raw ArrayBuffer
    |-- onnxParser       builds OnnxGraph (nodes, edges, params) from proto result
    |-- onnxruntime-web  InferenceSession.create() for WASM inference + benchmark
```

Key architectural decisions:
- The WASM binary files (ort-wasm-simd-threaded.*) are copied from node_modules to public/
  by a prebuild/predev npm script and served from the root URL. They are gitignored.
- The protobuf parser runs BEFORE InferenceSession.create() so the graph is shown even
  if the WASM session creation is slow or fails.
- ArrayBuffer is sliced before posting to the worker so both the parser and the session
  can read the same bytes (postMessage transfers destroy the sender's copy).

---

## What has been implemented

### v1 (bootstrap + MVP visualization)

Commits: e0e04d5 through 4e4c6c3 on master.

- Vite 8 + React 19 + TypeScript SPA scaffold
- vercel.json: SPA rewrite rules + COOP/COEP headers (required for SharedArrayBuffer)
- ModelDropzone: full-viewport drag-and-drop, FileReader -> ArrayBuffer, .onnx only
- GraphCanvas: React Flow + dagre top-down layout, custom OperatorNode + IONode
- LayerInspector: op type, param count, estimated size rows
- useOnnxWorker hook: LOAD_MODEL / RUN_INFERENCE / PROGRESS / ERROR protocol
- onnxWorker.ts: Web Worker with onnxruntime-web
- onnxParser.ts (v1): used unreliable session.handler?.model?.graph?.node internals
  - FALLBACK BUG: most models collapsed to a single "Model" node (fixed in v2)
- Theme: Avionics Blueprint CSS custom properties, JetBrains Mono, 4/8px grid
- WASM path fix: prebuild/predev script copies ort-wasm-simd-threaded.* to public/,
  ort.env.wasm.wasmPaths = '/' (production fix)
- Tests: 46 passing (graph.test.ts, onnx.test.ts, app.test.tsx)

### v2 (proper parser, sensitivity coloring, benchmark, Controls theme)

Branch: v2, merged via PR #1. Commits: 2caa159.

**Core fix: ONNX protobuf parser (src/lib/onnxProtoParser.ts)**
- Schema-aware binary protobuf decoder, no WASM internals dependency
- Decodes ModelProto -> GraphProto -> NodeProto / TensorProto / ValueInfoProto
- Wire types handled: VARINT (0), 64-bit (1), length-delimited (2), 32-bit (5)
- Extracts: node op_type / inputs / outputs, initializer name / dims / elemCount / sizeMB,
  graph input and output ValueInfoProto with tensor shapes
- Packed int64 dims handled (proto3 default for repeated int64)
- parseOnnxGraph now takes (buffer: ArrayBuffer, modelName: string) - no session needed
- Per-node param count = sum of elemCount for initializers whose name appears in node.inputs
- Tested against sageconv_Opset18.onnx (24 nodes, 20K params) and
  tagconv_Opset17.onnx (53 nodes, 40K params)

**Sensitivity coloring (GraphCanvas + LayerInspector)**
- OperatorNode border color based on paramCount:
  - < 100K:  rgba(255,255,255,0.15) (default)
  - 100K-1M: dim amber #8A7A00
  - 1M-10M:  orange #E67E22
  - > 10M:   red #C0392B
- LayerInspector shows SENSITIVITY row: MINIMAL / LOW / MEDIUM / HIGH with matching color

**Tensor shapes**
- Parsed from ValueInfoProto in GraphProto.input and GraphProto.output
- OnnxNode gained inputShapes?: OnnxDim[][] and outputShapes?: OnnxDim[][]
- OnnxDim = {value: number} | {param: string} (handles symbolic dims like "batch_size")
- formatShape() renders as "[1, 3, 224, 224]" or "[batch_size, 512]"
- Shown as small dim label on graph nodes and as labeled sections in LayerInspector

**Stats bar (App.tsx)**
- One-line bar above the canvas: model filename, total params, estimated size MB, node count
- Benchmark button: sends BENCHMARK command to worker (10 runs), shows avg/min/max ms
- Load new button: returns to the dropzone without page reload

**React Flow Controls theme (theme.css)**
- .react-flow__controls: dark surface (#16191C), 1px border rgba(255,255,255,0.12)
- .react-flow__controls-button: dark fill, amber on hover, no box-shadow

**useOnnxWorker additions**
- runBenchmark(runs): Promise<BenchmarkResult>
- benchmarkResult state: { avgMs, minMs, maxMs, runs } | null
- Status: added 'benchmarking' to the union type

**Tests: 60 passing** (up from 46)
- New: parseOnnxProto field decoding, formatShape, parseOnnxGraph topology/edges/params
- New: BENCHMARK_RESULT message handling, runBenchmark postMessage shape

---

## What is NOT yet implemented (roadmap)

### v3 scope (planned next branch)

Priority order based on user value:

1. **Dynamic INT8 quantization**
   - Challenge: onnxruntime-web is inference-only. Quantization requires modifying the
     ONNX protobuf (change weight initializer dtypes from FLOAT to INT8/UINT8, add
     DequantizeLinear ops, set scale/zero_point). Can be done by extending onnxProtoParser
     with a protobuf *writer* (encode modified graph back to binary).
   - Simpler first step: integer-rounding quantization (scale all float weights to INT8
     range, re-encode as FLOAT with reduced precision) that at least demonstrates the
     size change - real INT8 requires operator support in the runtime.
   - Vercel serverless alternative: Python function using onnxruntime.quantization -
     adds backend complexity and breaks the "zero install" pitch.

2. **Before/after comparison panel**
   - Once quantization produces a second ArrayBuffer, load both into separate workers
     and display: original size / quantized size, param count delta, output cosine
     similarity on a dummy input.

3. **Graph search and filter**
   - Search box to highlight nodes by op type or name
   - Filter to show only nodes above a parameter threshold

4. **Per-layer exclusion UI**
   - Checkbox in LayerInspector to mark a node as "excluded from quantization"
   - Persisted in a Set<string> in App state, passed to the quantization pass

5. **Model export**
   - Download the modified (quantized/pruned) ONNX as a .onnx file
   - Use Blob + URL.createObjectURL for the download trigger

6. **Improved shape inference**
   - Most models only expose ValueInfoProto shapes for graph-level inputs and outputs.
     Intermediate tensor shapes require running ONNX shape inference.
   - Can be approximated by tracking shapes through known ops (Conv, MatMul, etc.)
     using the initializer dims as hints.

---

## File map (current)

```
src/
  App.tsx                   root component, layout, state wiring
  components/
    GraphCanvas.tsx          React Flow canvas, dagre layout, sensitivity borders
    LayerInspector.tsx       selected node detail + shapes + sensitivity row
    ModelDropzone.tsx        full-viewport drop zone
  hooks/
    useOnnxWorker.ts         Worker lifecycle, loadModel, runBenchmark, runInference
  lib/
    onnxProtoParser.ts       binary ONNX protobuf parser (core of v2)
    onnxParser.ts            builds OnnxGraph from ParsedGraph + initializer sizes
    onnxTypes.ts             OnnxNode, OnnxEdge, OnnxGraph, OnnxDim interfaces
    graphUtils.ts            pure selection helpers: selectNode, deselectAll, validateEdges
  workers/
    onnxWorker.ts            LOAD_MODEL, RUN_INFERENCE, BENCHMARK, PROGRESS, ERROR
  styles/
    theme.css                CSS custom properties + React Flow Controls overrides
  __tests__/
    graph.test.ts            23 tests: selection model, edge validation
    onnx.test.ts             22 tests: proto parser, parseOnnxGraph, worker hook
    app.test.tsx             15 tests: integration, load flow, select, benchmark message
public/
  favicon.svg               Wireframe Tensor icon (v2+)
  ort-wasm-simd-threaded.*  WASM runtime files (gitignored, copied by prebuild script)
vercel.json                 SPA rewrite + COOP/COEP headers
vite.config.ts              Worker ES format, onnxruntime-web exclusion, Vitest config
package.json                copy-ort-wasm prebuild/predev scripts
```

---

## Test files (gitignored, in test_files/)

- sageconv_Opset18.onnx:   24 nodes, 3 initializers, 20K params (GNN)
- tagconv_Opset17.onnx:    53 nodes, 5 initializers, 40K params (GNN)
- gptneox_Opset17.onnx:    (GPT-NeoX, potentially large)
- adv_inception_v3_Opset16.onnx: (Inception v3, ~25M params)

Use adv_inception_v3 to test sensitivity coloring (should show many red high-param nodes).
Use sageconv or tagconv for fast iteration during development.

---

## Conventions

- No emojis, no em dashes, no "--" as sentence separator in any file or commit message
- No co-author tags in commits
- Font: JetBrains Mono only. No Inter, Roboto, or sans-serif.
- Spacing: strict 4px/8px grid. No arbitrary pixel values.
- Colors: #12161A base, #FFB000 amber, #4A5D23 military green, #C0392B error.
  No box-shadows, no gradients, no border-radius above 2px.
- Branches: one branch per version (v3, v4, ...). PR to master when complete.
- Tests: write tests alongside new code. Keep all existing tests passing.
- Build: npm run build must succeed (tsc -b + vite build) before any PR.
