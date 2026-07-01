# Forma: Development Status

This file is the canonical record of what has been built, how it works, and what comes next.
New sessions should read this before touching any code.

---

## Current state: v0.8.0 merged to master

Vercel deployment: https://forma-ml.vercel.app
GitHub: https://github.com/Hussain004/Forma
Branch strategy: one branch per version (v0.5, v0.6, ...), PR to master when done.
Releases: tagged on master and published as GitHub releases after each merge.

---

## Architecture overview

Forma is a fully browser-native SPA. There is no backend. Every operation runs in the browser.

```
App.tsx  (React 19, main thread)
  |-- ModelDropzone        drag-and-drop, FileReader -> ArrayBuffer
  |-- StatsBar             model name / params / size / node count / layout toggle / benchmark
  |-- GraphCanvas          React Flow + dagre layout, MiniMap, JumpController, hover tooltip
  |-- LayerInspector       selected node detail, multi-select aggregate, model summary histogram
  |
  useOnnxWorker hook       manages the Web Worker lifecycle
  |
  onnxWorker.ts (Web Worker, off-main-thread)
    |-- onnxProtoParser    binary protobuf parse of the raw ArrayBuffer
    |-- onnxParser         builds OnnxGraph (nodes, edges, params, graphInputs) from proto result
    |-- onnxruntime-web    InferenceSession.create() for WASM inference + benchmark + export
```

Key architectural decisions:
- The WASM binary files (ort-wasm-simd-threaded.*) are copied from node_modules to public/
  by a prebuild/predev npm script and served from the root URL. They are gitignored.
- The protobuf parser runs BEFORE InferenceSession.create() so the graph is shown even
  if the WASM session creation is slow or fails.
- ArrayBuffer is sliced before posting to the worker so both the parser and the session
  can read the same bytes (postMessage transfers destroy the sender's copy).
- useReactFlow() must be called inside a child of <ReactFlow>, not in the same component.
  Solution: JumpController component placed as a child inside the ReactFlow JSX tree.

---

## What has been implemented

### v0.1 (bootstrap + MVP visualization)

- Vite 8 + React 19 + TypeScript SPA scaffold
- vercel.json: SPA rewrite rules + COOP/COEP headers (required for SharedArrayBuffer)
- ModelDropzone: full-viewport drag-and-drop, FileReader -> ArrayBuffer, .onnx only
- GraphCanvas: React Flow + dagre top-down layout, custom OperatorNode + IONode
- LayerInspector: op type, param count, estimated size rows
- useOnnxWorker hook: LOAD_MODEL / RUN_INFERENCE / PROGRESS / ERROR protocol
- Theme: Avionics Blueprint CSS custom properties, JetBrains Mono, 4/8px grid
- WASM path fix: prebuild/predev script copies ort-wasm-simd-threaded.* to public/
- Tests: 46 passing (graph.test.ts, onnx.test.ts, app.test.tsx)

### v0.2 (proper parser, sensitivity coloring, benchmark)

- Schema-aware binary protobuf decoder (src/lib/onnxProtoParser.ts)
  - Decodes ModelProto -> GraphProto -> NodeProto / TensorProto / ValueInfoProto
  - Wire types: VARINT (0), 64-bit (1), length-delimited (2), 32-bit (5)
  - ParsedValueInfo includes elemType (ONNX data type: 1=float32, 7=int64, etc.)
- Sensitivity coloring: OperatorNode border color by paramCount tier
- Tensor shape annotations on graph nodes and in LayerInspector
- Stats bar: model filename, total params, size MB, node count
- Benchmark button: BENCHMARK worker command, shows avg/min/max ms
- React Flow Controls theme overrides in theme.css
- Tests: 60 passing

### v0.3 (filter, exclusion, INT8 estimate, export)

- Graph filter: search box dims non-matching nodes in real time
- Node exclusion: mark individual nodes as excluded; strikethrough styling applied
- INT8 size estimate: QUANTIZE_ESTIMATE message from worker, shown in stats bar and per-node
- Model export: EXPORT worker command, Blob + URL.createObjectURL download trigger
  - Filename strips original extension cleanly (model_export.onnx, not model.onnx_export.onnx)
- quantize.ts: estimateInt8Size(), compressionRatio(), formatQuantizeEstimate()
- Tests: ~90 passing (v3.test.ts added)

### v0.4 (INT8 estimate UI polish, download hardening)

- formatQuantizeEstimate() display helper with toFixed(1) rounding
- Export promise rejection propagated correctly from worker ERROR message
- Download button in stats bar with full lifecycle (postMessage -> EXPORT_RESULT -> Blob)
- Tests: ~110 passing (v4.test.ts added)

### v0.5 (MiniMap, jump-to-node, keyboard shortcuts, op histogram)

- MiniMap from @xyflow/react (amber nodes, dark mask)
- JumpController child component: Enter in filter field calls fitView() on first match
- Keyboard shortcuts: / focuses filter input, Escape clears filter and deselects node
- computeOpCounts(nodes): op type frequency map excluding Input/Output nodes
- Op type histogram in LayerInspector model summary (shown when no node selected)
- Tests: ~120 passing (v0.5.test.ts added)

### v0.6 (op category coloring, ancestor/descendant trace, graph depth)

- OP_CATEGORIES map: op type -> category string (Convolution, Activation, Normalization, etc.)
- opCategoryColor(opType): returns per-category color hex
- Left accent bar on OperatorNode uses category color instead of sensitivity border
- getAncestors(graph, nodeId): BFS backward through edges
- getDescendants(graph, nodeId): BFS forward through edges
- Trace mode: selecting a node highlights ancestors (blue) and descendants (green), dims rest
- computeGraphDepth(graph): longest path via topological sort
- DEPTH row in model summary histogram
- Category legend showing only categories present in the loaded model
- Tests: ~130 passing (v0.6.test.ts added)

### v0.7 (multi-select, aggregate inspector, hover tooltip)

- selectedNodeIds: Set<string> in App state
- Ctrl/Meta+click for multi-select via onNodeCtrlClick prop and setMultiSelection()
- bulkExclude(graph, ids) / bulkInclude(graph, ids) for batch operations
- Aggregate inspector: "N NODES SELECTED" with combined param count, size, op breakdown
- EXCLUDE ALL / INCLUDE ALL buttons in aggregate view
- Hover tooltip: onNodeMouseEnter/Leave, fixed-position overlay with op type, params, shape
- COPY button in LayerInspector: copies node metadata to clipboard via navigator.clipboard
- Tests: 144 passing across 9 files (v0.7.test.ts added)

### v0.8 (layout toggle, search dropdown, clipboard copy, benchmark type fix)

- Layout toggle: TB (top-down) / LR (left-right) button in stats bar, passed as layoutDir prop
- Search dropdown: up to 8 live-filtered results as you type; arrow keys navigate, Enter jumps,
  Escape dismisses; onMouseDown used instead of onClick to avoid blur-before-click race
- filterInputRef: useRef<HTMLInputElement> for / shortcut focus
- Benchmark tensor type fix: makeBenchmarkTensor() helper covers all ONNX data types
  - benchmarkInputTypes: Record<string, number> stored during LOAD_MODEL from graphInputs
  - Fixes crash on INT64-input models like GPT-NeoX
  - Note: onnxruntime-web uses 'float64' not 'double' as the tensor type string for elem type 11
- Isometric 3D favicon: three shaded faces (top/left/right) per slab, amber tones
- Tests: 144 passing across 9 files (v0.8.test.ts added)

---

## Planned next

### v0.9 (inspector depth)

- Attribute viewer: show op attributes (kernel size, strides, groups, etc.) in LayerInspector
  - Attributes are parsed in onnxProtoParser as a string dict; richer decoding needed
- Tensor shape annotations on graph edges: show tensor dimensions along the edges
- Search by tensor name in addition to op name

### v1.0 (power features)

- Subgraph collapse: fold a selected subtree into a single group node
- Side-by-side model diff: load two ONNX files, highlight structural differences
- Export filtered graph: strip excluded nodes and re-export a pruned ONNX file

---

## File map (current)

```
src/
  App.tsx                   root component, layout, state wiring, keyboard shortcuts
  components/
    GraphCanvas.tsx          React Flow canvas, dagre layout, MiniMap, JumpController,
                             hover tooltip, layout direction prop, trace role coloring
    LayerInspector.tsx       per-node detail, multi-select aggregate, model summary histogram,
                             COPY button, category swatches, DEPTH row
    ModelDropzone.tsx        full-viewport drop zone
  hooks/
    useOnnxWorker.ts         Worker lifecycle, loadModel, runBenchmark, exportModel
  lib/
    onnxProtoParser.ts       binary ONNX protobuf parser; ParsedValueInfo has elemType
    onnxParser.ts            builds OnnxGraph; returns graphInputs (non-initializer inputs)
    onnxTypes.ts             OnnxNode, OnnxEdge, OnnxGraph, OnnxDim, ParsedValueInfo
    graphUtils.ts            selection, filter, exclusion, tracing, depth, multi-select,
                             OP_CATEGORIES, opCategoryColor, computeOpCounts, computeGraphDepth
    quantize.ts              estimateInt8Size, compressionRatio, formatQuantizeEstimate
  workers/
    onnxWorker.ts            LOAD_MODEL, BENCHMARK, EXPORT, makeBenchmarkTensor
  styles/
    theme.css                CSS custom properties + React Flow Controls overrides
  __tests__/
    graph.test.ts            graph utilities, selection model, edge validation
    onnx.test.ts             proto parser, parseOnnxGraph, worker hook lifecycle
    app.test.tsx             integration: load flow, selection, error states
    v3.test.ts               filter, exclusion, INT8 estimation
    v4.test.ts               export reliability, quantize formatting, download filename
    v0.5.test.ts             computeOpCounts, keyboard shortcuts, op histogram
    v0.6.test.ts             opCategoryColor, getAncestors/getDescendants, computeGraphDepth
    v0.7.test.ts             setMultiSelection, bulkExclude/bulkInclude, aggregate inspector
    v0.8.test.ts             layout toggle, search dropdown, clipboard copy, benchmark types
public/
  favicon.svg               isometric 3D stacked layers (amber, three shaded faces per slab)
  ort-wasm-simd-threaded.*  WASM runtime files (gitignored, copied by prebuild script)
vercel.json                 SPA rewrite + COOP/COEP headers
vite.config.ts              Worker ES format, onnxruntime-web exclusion, Vitest config
package.json                copy-ort-wasm prebuild/predev scripts
```

---

## Conventions

- No emojis, no em dashes, no "--" as sentence separator in any file or commit message
- No co-author tags in commits
- Font: JetBrains Mono only. No Inter, Roboto, or sans-serif.
- Spacing: strict 4px/8px grid. No arbitrary pixel values.
- Colors: #12161A base, #16191C surface, #1C2128 raised, #FFB000 amber, #52C57A green,
  #C0392B error. No box-shadows, no gradients, no border-radius above 2px.
- Branches: one branch per version. PR to master when complete.
- Tests: write tests alongside new code. Keep all existing tests passing.
- Build: npm run build must succeed (tsc -b + vite build) before any PR.
- README: update version badge, test count, capabilities, and releases table on every version.
- GitHub releases: create immediately after each merge. Never let a merged version go untagged.
