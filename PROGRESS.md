# IMPROVEMENTS.md progress tracker

Tracks work against IMPROVEMENTS.md, one line per item. Updated as items land.
Branch: `ux-improvements`. Larger Initiatives are intentionally skipped -- each one
has an open question that needs an answer from the project owner before code should
be written (see IMPROVEMENTS.md's "Larger Initiatives" section for the options).

Legend: [ ] not started, [~] in progress, [x] done, [-] skipped (needs a decision)

## Quick Wins

- [x] P0 Real title and social meta tags
- [x] P0 Bundle a sample model
- [x] P0 Product identity on the empty state
- [x] P0 Humanize the malformed-file error
- [x] P1 Self-host JetBrains Mono
- [x] P1 Remove the dead reactflow 11 dependency and starter leftovers
- [x] P1 Rename "Download" and fix its filename
- [x] P1 Fix dim-text contrast and micro type sizes
- [x] P1 Make shortcuts discoverable
- [x] P1 Benchmark warmup and running state
- [x] P2 Layout toggle that says what it does
- [x] P2 Placement-mode polish
- [x] P2 MiniMap category colors and node-count grammar (model-name flex-width deferred to the stats bar responsive collapse item, where the bar's overflow behavior gets rebuilt properly)
- [x] P2 One-prop render culling for big graphs (gated on `onnxNodes.length > 300`, not always-on -- jsdom reports zero-size bounding rects in tests, which would cull every node in every small fixture if this ran unconditionally)
- [x] P2 Consolidate the styling system
- [x] P2 Free-text Add Node input count

## Medium Effort

- [x] P0 Decouple graph rendering from WASM session creation
- [x] P0 Stop operation errors from destroying the workspace
- [x] P0 An annunciator line for silent actions
- [x] P1 Accept a dropped model anytime
- [ ] P1 Motion system: CSS only, glide not bounce
- [ ] P1 Layer Inspector information architecture
- [ ] P1 Edge insert needs intent
- [ ] P1 Stats bar overflow strategy
- [ ] P1 Keyboard and ARIA pass on the editing surface
- [ ] P2 Box select

## Larger Initiatives (blocked on a decision, not started)

- [-] P0 The exported-model validity story -- needs a call on disclaimer vs arity/dtype checks vs export-verify-roundtrip
- [-] P1 Big-model strategy -- needs a call on the supported node-count ceiling and what happens above it
- [-] P1 Desktop-only gate versus mobile support -- needs a call on gate-only vs a real mobile viewer
- [-] P2 Command palette -- needs a call on whether it becomes the unifying interaction
- [-] P2 How much onboarding is too much -- needs a call on passive-only vs one-time hint chips

## Log

(most recent first)

- A model can now be dropped onto the app at any time to replace the one currently open, not just from the initial empty state -- previously the only way back to a file picker was the "Load new" button. Dragging a file over the loaded workspace shows a full-viewport "Drop to replace current model" overlay (with an "Unsaved edits will be lost" warning when there are pending edits), guarded by a depth counter so dragenter/dragleave flicker between child elements doesn't flash the overlay on and off. Verified in a real browser (not just jsdom) by dispatching native DragEvents: overlay appears on dragenter, the model actually swaps on drop (node count went 3 to 8, confirming tiny.onnx was replaced by the sample model), overlay clears on dragleave without loading anything.
- Implemented together since one needs the other: worker ERROR messages now carry a `scope: 'load' | 'operation'` tag (LOAD_MODEL failures vs everything else). Load errors keep the existing full-screen dropzone/error behavior; operation errors (a failed benchmark, export, or inference) now keep the loaded graph mounted and status at 'ready' instead of tearing the whole workspace down for what might just be a transient WASM hiccup. A new always-present 22px status strip under the StatsBar (an "annunciator line," not a floating toast, per the audit's explicit design note) surfaces operation errors plus several previously-silent actions: rejected rewires (with the real reason -- self-connect, cycle, wrong node type), bulk-delete skip counts when nodes were ambiguous/ineligible, undo confirmations naming what was undone, and a copy confirmation. Verified live: self-connecting a node's output to its own input shows "Rewire rejected: Cannot connect a node to itself", Ctrl+Z after an attribute edit shows "Undid kernel_shape edit", and COPY shows "Copied to clipboard" -- all in the strip, with the graph staying fully interactive throughout.

- The single highest-impact fix in the whole list: `onnxWorker.ts` used to post `MODEL_LOADED` only after `ort.InferenceSession.create` resolved, so first paint of the graph waited on a full WASM runtime download+compile even though the graph itself was already parsed. The original live-site audit measured this at roughly 60 seconds over a real network. Session creation is now lazy (`ensureSession()`), triggered only by the first Benchmark click or inference call, from the same retained `exportBuffer` bytes. Verified live: graph now renders in ~120ms after picking a file (session doesn't exist yet), Benchmark still works correctly on first click (lazy creation succeeds, warmup+median intact), and switching to a second model correctly resets and recreates the session rather than reusing a stale one (added an explicit `session = null` reset in the LOAD_MODEL handler, since the module-level session variable used to only ever get reassigned, never cleared).

- Replaced the static og-image.png mockup (built before a sample model existed) with a real screenshot of the app running the actual bundled sample model -- closes the gap noted in the very first log entry.

- Bundled a small (1.2 KB) hand-encoded sample model at public/sample-model.onnx -- a real, runnable 6-node conv classifier (Conv -> Relu -> MaxPool -> Reshape -> Gemm -> Softmax) chosen specifically to hit five distinct op categories at once, so the first thing a new visitor loads showcases the category-color legend instead of a single gray/green node. Verified it both loads and actually runs through onnxruntime-web's Benchmark, not just parses. The empty state now also has a FORMA wordmark, a one-line pitch (the "never leaves this device" privacy claim only became literally true after the font self-hosting fix earlier), a "Load sample model" button wired to fetch + the same onModelLoaded path as a real drop, and a GitHub link -- previously a first-time visitor with no .onnx file handy had zero path forward and would just bounce off a bare "drop .onnx or .tflite model" line.

- Added a `?`-toggled keyboard shortcuts overlay listing all seven existing shortcuts (`/`, click, Ctrl+Click, drag-to-rewire, Delete, Ctrl+Z, Esc), closable via Esc or a Close button. Filter input placeholder now reads "FILTER NODES  /" with a title attribute, and the default Layer Inspector panel (Model Summary, shown whenever no node is selected) gets a "Click a node to inspect. Press ? for shortcuts." hint line. None of these shortcuts were new -- they existed since earlier versions but had zero in-app discoverability.
- Benchmark now runs 2 untimed warmup iterations before the 10 timed runs (the first run after a model loads pays for JIT/allocator warmup that skewed avg/max), reports median alongside avg/min/max, and annotates the label with "batch 1, zeroed inputs" so the numbers aren't mistaken for something more rigorous than they are. Button disables and reads "Running" while in flight. New deterministic test (the tiny fixture completes benchmarking too fast for a wall-clock Playwright check to ever catch the in-flight state, so verified via a mock-worker unit test instead of live).

- Merged the two overlapping `.react-flow__controls` CSS blocks (index.css's was silently dead -- theme.css's `!important` version always won the cascade, and was also the only one with the hover-to-amber and last-child rules) into a single token-based block in theme.css, and deleted the now-redundant inline `style` prop on `<Controls>` that duplicated it. Swept hardcoded hex colors that duplicate existing tokens (#FFB000, #16191C, #1C2128, #12161A, #E8EAF0, #8A8F9E) across GraphCanvas.tsx, App.tsx, LayerInspector.tsx, and ModelDropzone.tsx onto `var(--color-amber)` / `var(--bg-*)` / `var(--text-*)`. Deliberately left three spots alone since they're their own semantic tier scales that happen to reuse the same hex values, not simple token substitutions: `sensitivityColor()`, `CATEGORY_LEGEND`, and `traceAccent()` -- coupling those to the general-purpose tokens would mean an unrelated token change silently reshuffles those scales. Verified live with no broken `var()` resolution.

- Vendored the JetBrains Mono latin subset (weights 300/400/500/700) into public/fonts/ and replaced the Google Fonts @import with local @font-face rules. Sourced the files from the @fontsource/jetbrains-mono npm package (proper licensing, no manual font wrangling), then uninstalled the package itself since only the static files were needed. Verified live: zero external hosts contacted on load (previously fonts.googleapis.com and fonts.gstatic.com), which makes the "your model never leaves the browser" pitch literally true.

- Malformed-file error now shows a short friendly headline ("This file doesn't look like a valid ONNX or TFLite model.") with the raw onnxruntime message demoted to a smaller, non-shouting sentence-case line underneath, instead of dumping the full ALL-CAPS ORT error as the primary message. Verified live against a real junk file.

- "Download" (which returns unmodified original bytes) renamed to "Download Original" with a clarifying title, and its filename suffix fixed from the misleading `_export.onnx` to `_original.onnx`. "Export Modified" now shows a live edit count, e.g. "Export Modified (3)", from `attrOverrides.size + structuralOps.length`.

- Raised `--text-dim` from #4A4F5E (~2.2:1, WCAG fail) to #7A8191 (~4.5:1, WCAG AA pass) and pointed the hardcoded #5A6070 shape-label color and #3A4050 idle-pencil color at the same token instead of their own separate low-contrast values. Bumped MOD/NEW node badges from 7px to 9px and shape labels from 9px to 10px. Verified live: node shape labels ([1, 4]) went from barely visible to clearly legible.

- Placement ghost now scales with canvas zoom (via `onViewportChange`, CSS transform: scale from center) instead of staying a fixed screen size while the real node scales with the viewport; added an "ESC TO CANCEL" hint. Verified live at a zoomed-out viewport (screenshot: ghost visibly smaller than the real node behind it).

- Free-text Add Node entry gets an input-count stepper (1-8, defaulting to 1) instead of hardcoding a single input; stepper buttons use preventDefault on mousedown so they don't blur the text field and close the dropdown. New test + verified live.

- Layout toggle now reads "LAYOUT TB" / "LAYOUT LR" with a title attribute; MiniMap nodes colored by op category instead of uniform amber; node count singularizes to "1 NODE"; ReactFlow gets `onlyRenderVisibleElements` gated behind a 300-node threshold. Verified live in a production preview build (screenshot: minimap shows the Relu node green, matching its Activation category).

- Removed the unused `reactflow` v11 package (nothing imports it, `@xyflow/react` v12 is the real dependency) and Vite-starter leftovers (react.svg, vite.svg, empty App.css). Left hero.png alone since it isn't referenced anywhere and wasn't called out in the audit -- may be intentional for future README use.

- Real title + OG/Twitter meta tags in index.html; generated public/og-image.png (a static Avionics-Blueprint-styled card, not a live screenshot, since no sample model existed yet at this point in the work).
