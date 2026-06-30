# Forma: A Visual Editor for Neural Network Models

> *"Netron lets you see your model. Forma lets you shape it."*

A local-first, open-source GUI that lets you load a trained model, see it laid out as an interactive graph, and directly quantize, prune, and benchmark it (with live size/accuracy feedback) without writing a config file or a line of Python.

---

## 1. The Problem This Solves

Every serious model-compression tool today is code-first:

| Tool | What it does | Interface |
|---|---|---|
| ONNX Runtime quantization | INT8 dynamic/static PTQ | Python API |
| Microsoft Olive | Multi-pass optimization pipelines | JSON config + CLI |
| Microsoft NNI | Pruning, quantization, NAS | Python scripts |
| NVIDIA Model-Optimizer | Quantization, sparsity, distillation | Python SDK |
| PyTorch `torch.ao.quantization` | PTQ/QAT | Python API |

And the one popular **visual** tool, **Netron**, is read-only where you can inspect a graph's structure and shapes, but you can't act on what you see.

Nobody has merged "see the model" with "edit the model." That's the gap. Forma is the tool that goes from *inspection* to *intervention*: click a layer, quantize it, see the size and accuracy impact immediately, export the result.

**Target users:** ML engineers deploying to edge/mobile/browser, indie developers shipping local-AI features, students learning what quantization actually does to a network instead of treating it as a black-box CLI flag.

---

## 2. Core Concept

```
Load model → Visualize graph → Select layer(s) → Choose operation → 
Apply (runs locally) → See before/after (size, latency, accuracy proxy) → 
Export
```

The key design principle, borrowed from what made BAGEL work: **zero friction**. No installing CUDA toolkits to try it, no editing YAML. Drag a file in, get a usable result in under a minute for the simple case (whole-model INT8 quantization), with progressively deeper controls available for power users (per-layer mixed precision, structured pruning, sensitivity analysis).

---

## 3. Feature Scope

### MVP (what "v1.0" needs to do)
- Load an `.onnx` model
- Render the computation graph as an interactive, zoomable node diagram
- Per-node inspector: op type, parameter count, estimated size (MB), input/output shapes
- Whole-model **dynamic INT8 quantization** (one click, no calibration data needed — this is the easiest win and uses `onnxruntime.quantization.quantize_dynamic`)
- Whole-model **static INT8 quantization** (requires a small calibration dataset the user uploads — a folder of 20-100 representative inputs)
- Before/after comparison panel: file size, parameter count, and (if calibration data was provided) output similarity score
- Local benchmark: measure inference latency before and after, on the user's actual hardware, via `onnxruntime.InferenceSession`
- Export the modified model as `.onnx`

### v2 (what makes it genuinely differentiated, not just "quantize_dynamic with a GUI")
- **Per-layer control**: select a subset of layers and exclude them from quantization (maps to ONNX Runtime's `nodes_to_exclude` parameter) — useful because some layers (e.g. the first/last layers of a network) are disproportionately sensitive to precision loss
- **Sensitivity analysis mode**: a one-shot pass that estimates which layers are most sensitive to quantization (using weight-distribution heuristics or a fast Hessian-trace approximation), so the GUI can color-code layers green/yellow/red *before* the user even decides what to quantize
- **Structured pruning**: channel/filter pruning via `torch.nn.utils.prune`, with a slider for target sparsity and a live-updating size estimate
- **Mixed-precision search**: given a target model size or a target accuracy floor, auto-suggest a per-layer bit-width assignment (this is the same idea as NVIDIA Model-Optimizer's format search, but exposed visually instead of as a search-space config)
- **PyTorch input support**: accept `.pt`/`.safetensors`, convert to ONNX as a first step via `torch.onnx.export`, then proceed through the same pipeline

### v3 (stretch / ecosystem plays)
- GGUF export path for LLM-specific quantization (GPTQ/AWQ-style), since the local-LLM community cares deeply about this
- Quantization-aware fine-tuning (QAT) loop for cases where PTQ accuracy loss is too large — this is the step that actually needs your GPU at *your* development time too, not just the end user's
- Plugin system so others can add custom compression passes
- "Diff" mode: load two versions of a model side by side and visually diff the graphs

---

## 4. Tech Stack

**Frontend** — reuse your BAGEL playbook, since it already proved this works:
- React + TypeScript + Vite
- `react-flow` (or `elkjs` for auto-layout) for the graph visualization — this is the single most important UI component, worth prototyping first
- TailwindCSS for styling
- A charting library (`recharts` or `visx`) for the before/after size and accuracy charts

**Backend** — Python, because every compression library worth using is Python-first:
- FastAPI serving a local REST + WebSocket API (WebSocket specifically for streaming quantization/calibration progress — these jobs can take seconds to minutes on larger models, and a frozen UI with no feedback is a fast way to lose users)
- `onnx` and `onnxruntime` for graph parsing and the actual quantization engine
- `onnxruntime.quantization` for dynamic/static PTQ
- `torch` + `torch.ao.quantization` / `torch.nn.utils.prune` for the PyTorch-native pruning path
- Optionally shell out to Microsoft Olive as an advanced backend for multi-pass LLM workflows in v3, rather than reimplementing its optimization passes yourself

**Packaging** — this matters as much as the features, since "zero install" is the whole pitch:
- Tauri (preferred over Electron — smaller binary, Rust-based, faster) wrapping the React frontend
- Bundle the Python backend with PyInstaller, or ship a one-line setup script (`pip install -r requirements.txt && python server.py`) for the open-source/dev version, with a packaged installer as a later nicety
- Look at how IOPaint and Upscayl handle this — both are good reference points for "polished one-click installer wrapping a Python ML backend"

---

## 5. Architecture

```
┌──────────────────────────────────────────────┐
│  Browser (React + react-flow)                │
│  - Graph renderer                            │
│  - Layer inspector panel                     │
│  - Operation controls (quantize/prune)       │
│  - Before/after comparison charts            │
└───────────────────┬──────────────────────────┘
                     │ REST (load/export) + WebSocket (job progress)
┌───────────────────▼───────────────────────────┐
│  Local backend (FastAPI, Python)              │
│  ┌─────────────┐ ┌──────────────┐             │
│  │Model loader │ │ Job runner   │             │
│  │(onnx parse) │ │ (async, WS)  │             │
│  └─────────────┘ └──────────────┘             │
│  ┌─────────────┐ ┌──────────────┐             │
│  │Quantization │ │ Benchmark    │             │
│  │engine       │ │ runner       │             │
│  └─────────────┘ └──────────────┘             │
└───────────────────┬───────────────────────────┘
                    │
              ┌─────▼─────────┐
              │  Filesystem   │
              │ (model file   │
              │cache,exports) │
              └───────────────┘
```

Suggested module breakdown (mirrors what a clean BAGEL-style repo would look like):

```
Forma/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── GraphCanvas.tsx       # react-flow graph rendering
│   │   │   ├── LayerInspector.tsx    # right-panel node details
│   │   │   ├── OperationPanel.tsx    # quantize/prune controls
│   │   │   ├── ComparisonChart.tsx   # before/after visualization
│   │   │   └── JobProgress.tsx       # WebSocket-driven progress bar
│   │   ├── hooks/
│   │   │   └── useModelGraph.ts      # fetch + parse graph from backend
│   │   └── lib/
│   │       └── websocket.ts
├── backend/
│   ├── main.py                       # FastAPI app entrypoint
│   ├── model_loader.py               # onnx.load, graph → JSON for frontend
│   ├── quantize.py                   # wraps onnxruntime.quantization
│   ├── prune.py                      # wraps torch.nn.utils.prune
│   ├── sensitivity.py                # per-layer sensitivity heuristics
│   ├── benchmark.py                  # latency measurement
│   └── jobs.py                       # async job queue + WS progress events
└── README.md
```

---

## 6. UX Flow (step by step)

1. **Drop a model file** onto the app (`.onnx` to start)
2. Backend parses the graph (`onnx.load` + `onnx.shape_inference`), returns a JSON node/edge list with per-node metadata (op type, param count, size)
3. Frontend renders this with `react-flow`. Large models (1000+ nodes, e.g. transformer blocks) get auto-collapsed into representative "block" groups that expand on click — this is the single hardest UI problem here, worth solving early (see Section 7)
4. User clicks a node or marquee-selects a region → right panel shows details and available operations
5. User picks **"Quantize selected"** or **"Quantize entire model"**
6. If static quantization is chosen, the UI prompts for a small calibration folder (sample inputs); dynamic quantization skips this
7. Click **Apply** → job runs server-side, progress streams over WebSocket, frontend shows a live progress bar
8. **Comparison panel** updates: original size vs. new size, parameter count delta, and (if calibration data was given) an output-similarity score computed by running both models on the calibration set and comparing logits/embeddings via cosine similarity
9. **Benchmark button**: runs N inference passes on both versions using `onnxruntime.InferenceSession` on the user's own hardware, reports latency before/after
10. **Export** → download the modified `.onnx` file

---

## 7. The Hard Technical Problems (and how to approach them)

**Graph rendering performance on large models.**
A ResNet-50 has ~175 nodes — easy. A transformer-based vision model can have thousands. Naively rendering every node in `react-flow` will choke the browser. Solution: detect repeated structural patterns (e.g., the same transformer block repeated 24 times) and collapse them into a single representative node by default, with a "expand all instances" toggle. This is conceptually similar to how point-cloud viewers downsample for display while keeping full data underneath — render a simplified view, keep the full graph in memory for operations.

**Giving meaningful accuracy feedback without requiring a full eval harness.**
Most users won't have a labeled validation set sitting around. The pragmatic answer: use cosine similarity between original and quantized model outputs on whatever calibration data the user provides as a *fast proxy* — not a replacement for real evaluation, but useful enough to catch "this destroyed the model" cases instantly. Be explicit in the UI that this is a proxy, not a guarantee, with an option for power users to plug in a custom eval script that returns a real accuracy number.

**PyTorch input support is fragile.**
`torch.fx` graph capture breaks on models with dynamic control flow (if/else branches, loops based on tensor values), which is common in anything beyond standard CNNs/transformers. Scope the MVP to ONNX-only — it's also the more useful target anyway, since it's what most deployment runtimes (mobile, browser, edge) actually consume. Offer PyTorch input only as "convert to ONNX first" using `torch.onnx.export`, and clearly document which architectures that does and doesn't handle well.

**Per-layer sensitivity analysis is expensive if done naively.**
The "correct" way (re-run full evaluation with each layer individually quantized) doesn't scale. Use a cheap proxy for the interactive view — weight magnitude distribution and activation range statistics correlate reasonably well with quantization sensitivity — and offer a slower, more rigorous "deep analysis" mode as an explicit opt-in for users willing to wait.

---

## 8. Build Plan (phased)

| Phase | Scope | Why this order |
|---|---|---|
| 1 | ONNX parsing + static graph visualization | De-risk the hardest UI problem (graph rendering) before building anything else on top of it |
| 2 | Whole-model dynamic INT8 quantization + size comparison | Smallest possible end-to-end slice — proves the full pipeline works |
| 3 | Static quantization with calibration upload + latency benchmark | Adds the "real" quantization path and the local hardware benchmarking story |
| 4 | Per-layer selection + exclude-from-quantization controls | This is what differentiates Forma from "quantize_dynamic with extra steps" |
| 5 | Pruning support (PyTorch path) + sensitivity coloring | Second compression technique, plus the visual sensitivity analysis that makes the tool feel genuinely smart |
| 6 | Packaging (Tauri/installer), README, demo GIF, docs | Polish pass — this is what actually drives adoption once the tool works |

Build the demo GIF around the most satisfying moment: a model going from, say, 90MB/94% to 23MB/93.1%, shown as a live, interactive before/after — that single clip is your launch post on r/MachineLearning, r/LocalLLaMA, and Hacker News.

---

## 9. What to Lean On Instead of Reinventing

- **Netron** — not as a dependency, but as a UX reference for how to render ONNX graphs clearly. Don't copy its code; learn from its layout decisions.
- **`onnxruntime.quantization`** — this is the actual quantization engine. Forma's value is the interface and workflow around it, not a competing quantization implementation.
- **Microsoft Olive** — a reasonable v3 backend option for advanced multi-pass LLM optimization pipelines, rather than reimplementing those passes.
- **Your own BAGEL architecture** — the drag-and-drop-first UX, the "useful in under a minute" philosophy, and the clean separation between a fast frontend and a focused backend already worked once. Reuse the playbook.

---

## 10. Honest Risks

- The accuracy "proxy" (output similarity on calibration data) is not a real accuracy benchmark. Don't let the UI imply more confidence than the metric deserves — a misleading green checkmark on a broken model would undermine trust fast.
- PyTorch support will always be partial given dynamic graphs. Be upfront about this rather than promising universal compatibility.
- Competing with Netron's enormous format support (ONNX, TensorFlow, Keras, Core ML, Caffe, and a dozen more) is not realistic short-term. Win on ONNX depth first; expand format support only if there's real demand.

---
