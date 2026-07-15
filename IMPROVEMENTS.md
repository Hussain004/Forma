# Forma UI/UX Improvement Plan

Grounded in a full read of the source (App.tsx, GraphCanvas.tsx, LayerInspector.tsx,
ModelDropzone.tsx, useOnnxWorker.ts, onnxWorker.ts, index.css, theme.css, index.html,
package.json) plus a scripted Playwright session against the live app at
forma-ml.vercel.app on 2026-07-15. Every claim below was verified in code or observed
live; nothing is guessed.

---

## The three critiques

### 1. The skeptical HN commenter

- "The tab says `forma-app` and the link preview on HN is blank. Zero meta tags. This
  is a Vite starter someone forgot to rename."
- "It's pitched as fully client-side, but the first thing it does is call
  fonts.googleapis.com and fonts.gstatic.com. If the pitch is 'your model never leaves
  the browser', the network tab should be empty."
- "I don't have an ONNX file on my phone or work laptop. There's no sample model, so I
  saw a black page with one sentence and closed the tab. You lost 80 percent of your
  traffic at the door."
- "I clicked an edge to see what it does and it silently inserted an Identity node into
  my model. An editor that mutates the graph on a single unlabeled click is not an
  editor I trust to produce a valid export."
- "package.json has both `reactflow` 11 and `@xyflow/react` 12. One of them is dead
  weight. If the dependency list is sloppy, what does the protobuf writer look like?"
- "I rewired a Conv output into a MatMul input and it exported without a word. Does
  this thing validate anything, or will it happily hand me a broken model?"
- "Benchmark with no warmup run and zero-filled inputs, reporting avg of 10 runs
  including the first JIT-cold one? Those numbers are noise."

### 2. First 60 seconds for a r/MachineLearning / r/webdev visitor

0 to 5 s: a near-black page, a small crosshair, and "DROP .ONNX OR .TFLITE MODEL" in
dim caps. No product name, no explanation, no GitHub link, no sample. The visitor who
has a model handy continues; everyone else is gone.

5 to 30 s: they drop a model. The progress bar jumps to "LOADING WASM RUNTIME" and
sits there. In the audited session the graph did not appear for roughly 60 seconds
because MODEL_LOADED is not posted until onnxruntime's ~20 MB WASM session finishes
downloading and compiling, even though the graph itself was parsed in milliseconds
(onnxWorker.ts posts MODEL_LOADED after `InferenceSession.create`). The visitor
assumes the tool is broken. This is the single worst moment in the product.

30 to 60 s: the graph appears and genuinely looks good; the Avionics Blueprint reads
as intentional. They click a node, the inspector fills in, category colors make sense.
But nothing tells them they can edit an attribute (a pencil appears only on hover),
nothing hints at drag-to-rewire, `/`, Delete, or Ctrl+Z, and if they click an edge
they just silently edited the model without knowing it.

### 3. Heuristic evaluation (Nielsen)

| Heuristic | Verdict | Worst offender |
|---|---|---|
| Visibility of system status | Fail | Invalid rewire drops, skipped bulk deletes, and COPY are all silent; progress percent is fake (fixed 10/50/100) |
| Match with the real world | Weak | "TB" / "LR" button, "SENSITIVITY" (it is only a param-count bucket), "Download" that returns the original file named `_export.onnx` |
| User control and freedom | Weak | No redo (v1.6 planned); a failed benchmark or export flips global status to `error` and replaces the whole workspace with the full-screen error dropzone |
| Consistency and standards | Weak | Two parallel React Flow control override blocks (index.css and theme.css, the latter with !important); hardcoded hex values bypassing the tokens across all components |
| Error prevention | Fail | Plain click on an edge inserts a node (verified live: node count 3 to 4); no confirmation, no announcement |
| Recognition over recall | Fail | Every shortcut ( / , Delete, Esc, Ctrl+Z, ctrl-click multi-select, drag-to-rewire) is invisible until discovered by accident |
| Flexibility and efficiency | OK | Shortcuts exist and are good; they are just undiscoverable; no box-select |
| Aesthetic and minimalist design | Pass | The strongest axis; the system is coherent and restrained |
| Help users recognize and recover from errors | Fail | Malformed file shows raw ORT output: "CAN'T CREATE A SESSION. ERROR_CODE: 7, ERROR_MESSAGE: FAILED TO LOAD MODEL BECAUSE PROTOBUF PARSING FAILED." (verified live) |
| Help and documentation | Fail | None in-app, not even a GitHub link |

---

## Quick Wins (implementable in under a day)

### [Priority: P0] Real title and social meta tags
- **Problem:** `<title>forma-app</title>`, zero description/OG/Twitter tags (verified live). A Show HN link renders with no preview and the Vite default project name. This is the cheapest possible credibility loss.
- **Fix:** Title "Forma: ONNX and TFLite visualizer and editor in the browser"; add meta description, og:title, og:description, og:image (a static screenshot in public/), twitter:card.
- **Files affected:** index.html, public/ (one og-image.png)
- **Implementation notes:** Pure HTML. Take the og:image from the loaded-graph state, not the empty state.
- **Effort:** S

### [Priority: P0] Bundle a sample model
- **Problem:** The empty state assumes the visitor has an .onnx file within reach. Most HN/Reddit traffic does not; they bounce off a black page. This is the highest-leverage conversion fix in the entire document.
- **Fix:** Ship MNIST (about 26 KB from the ONNX model zoo) in public/ and add a "LOAD SAMPLE MODEL" button under the drop prompt that fetches it and calls the existing `onModelLoaded` path.
- **Files affected:** ModelDropzone.tsx, public/mnist.onnx
- **Implementation notes:** `fetch('/mnist.onnx').then(r => r.arrayBuffer())` into the same code path as a dropped file, so the worker/tests see no new surface. One new test in the app suite. Keep the sample small so the repo stays light.
- **Effort:** S

### [Priority: P0] Product identity on the empty state
- **Problem:** No wordmark, no one-line value prop, no GitHub link anywhere in the app (verified: body text is literally "DROP .ONNX OR .TFLITE MODEL"). A Show HN visitor who likes the tool has no path to the repo, which is the entire point of launching.
- **Fix:** Add to the dropzone: "FORMA" wordmark, one line ("Visualize, edit, and re-export ONNX models. Runs entirely in your browser; your model never leaves this device."), a GitHub link, and the sample-model button from the item above. Keep it to four quiet lines; the austerity is an asset.
- **Files affected:** ModelDropzone.tsx
- **Implementation notes:** The privacy line is the HN hook; it is only honest once the Google Fonts call is removed (see below). Also add a small GitHub link at the far right of StatsBar so it survives past load.
- **Effort:** S

### [Priority: P0] Humanize the malformed-file error
- **Problem:** A junk file yields raw onnxruntime output in full-screen uppercase (verified live). Nielsen: error recovery. It also reveals that parse errors surface from session creation rather than your own parser.
- **Fix:** In the worker catch, map known failure shapes to "This file doesn't look like a valid ONNX or TFLite model." with the raw message in a smaller, dim, non-uppercased second line. Remove `text-transform: uppercase` from the error span so raw messages stop shouting.
- **Files affected:** src/workers/onnxWorker.ts, ModelDropzone.tsx
- **Implementation notes:** Message-prefix mapping, no library. Existing error-path tests in the app suite assert on error display; update the asserted strings.
- **Effort:** S

### [Priority: P1] Self-host JetBrains Mono
- **Problem:** theme.css imports Google Fonts (verified live: fonts.googleapis.com and fonts.gstatic.com are contacted). It contradicts the client-side privacy pitch, is render-blocking, and breaks offline use. HN checks the network tab on tools like this.
- **Fix:** Vendor the four woff2 weights into public/fonts/ with @font-face declarations, `font-display: swap`. Delete the @import.
- **Files affected:** src/styles/theme.css, public/fonts/
- **Implementation notes:** No new dependency needed; @fontsource/jetbrains-mono is fine too if you prefer npm-managed files. Zero test risk. After this, "zero external requests" is a true sentence you can say in the HN post.
- **Effort:** S

### [Priority: P1] Remove the dead reactflow 11 dependency and starter leftovers
- **Problem:** package.json carries both `reactflow` ^11 and `@xyflow/react` ^12; grep confirms nothing imports `reactflow`. src/assets/react.svg, vite.svg, and the one-line App.css are Vite-starter residue. Repo readers on HN notice this kind of thing.
- **Fix:** `npm uninstall reactflow`; delete the unused assets and App.css if nothing imports them.
- **Files affected:** package.json, src/App.css, src/assets/
- **Implementation notes:** Verify with a build and full test run; zero expected risk.
- **Effort:** S

### [Priority: P1] Rename "Download" and fix its filename
- **Problem:** "Download" returns the original untouched bytes but names the file `<model>_export.onnx`, which implies edits were applied. Next to "Export Modified" the distinction is invisible. (Nielsen: match with the real world.)
- **Fix:** Rename the button to "Original" or "Download original"; name the file `<model>_original.onnx`. Show an edit count on the primary action: "EXPORT MODIFIED (3)" from `attrOverrides.size + structuralOps.length`.
- **Files affected:** src/App.tsx
- **Implementation notes:** Any test using getByText('Download') (v1.1 era) needs its string updated. The count badge is a two-line change and doubles as pending-work visibility until the v1.6 history panel lands.
- **Effort:** S

### [Priority: P1] Fix dim-text contrast and micro type sizes
- **Problem:** Measured ratios: `--text-dim` #4A4F5E is 2.2:1 on the base background and it is used for real content (section headers, INT8 estimates, TFLite badge, hints). Node shape labels (#5A6070, 9 px) are 2.8:1. The MOD/NEW badges are 7 px type. All fail WCAG 1.4.3 and are genuinely hard to read on a normal-brightness display.
- **Fix:** Raise `--text-dim` to #7A8191 (about 4.7:1 on both backgrounds, still clearly a third tier under `--text-secondary`). Minimum 9 px for badge text, 10 px for shape labels. Keep the current darker value as a `--text-faint` token for purely decorative strokes if wanted.
- **Files affected:** src/styles/theme.css, GraphCanvas.tsx (hardcoded #5A6070 instances), App.tsx
- **Implementation notes:** Pure token change plus a size bump; no test risk. This is the one accessibility fix that also visibly improves the aesthetic at a glance.
- **Effort:** S

### [Priority: P1] Make shortcuts discoverable
- **Problem:** `/` focus-filter, Delete, Esc, Ctrl+Z, ctrl-click multi-select, and drag-to-rewire all exist and all are invisible (Nielsen: recognition over recall). Power-user features that nobody finds are features you did not ship.
- **Fix:** (1) Filter placeholder becomes "FILTER NODES  /" and gets a title attribute. (2) A `?` key toggles a small static overlay panel listing the shortcuts in the established row style. (3) A one-line hint in the inspector empty state: "Click a node to inspect. Press ? for shortcuts."
- **Files affected:** src/App.tsx (keydown handler, StatsBar placeholder), a small ShortcutsOverlay component
- **Implementation notes:** Static div, Escape closes, aria role="dialog". Reuses row/label styles from LayerInspector. Add one keyboard test.
- **Effort:** S

### [Priority: P1] Benchmark warmup and running state
- **Problem:** The 10 timed runs include the first cold run (JIT and memory-allocation heavy), skewing avg and max; there is no warmup in onnxWorker.ts. The button also stays clickable while a run is in flight. HN will screenshot dubious numbers.
- **Fix:** Run 2 untimed warmup iterations before the timed loop; report median alongside avg. Disable the button and label it "RUNNING" while status is benchmarking. Optionally annotate the result label with "batch 1, zeroed inputs".
- **Files affected:** src/workers/onnxWorker.ts, src/App.tsx
- **Implementation notes:** Worker-side benchmark tests (if any assert run counts) need the warmup accounted for.
- **Effort:** S

### [Priority: P2] Layout toggle that says what it does
- **Problem:** A bare "TB" button is unreadable to anyone who has not internalized dagre's rankdir values, and it is ambiguous whether it shows current state or the target.
- **Fix:** Label it "LAYOUT TB" / "LAYOUT LR" with a title attribute "Toggle top-bottom / left-right layout".
- **Files affected:** src/App.tsx (StatsBar)
- **Effort:** S

### [Priority: P2] Placement-mode polish
- **Problem:** The ghost says "CLICK TO PLACE" but not that Esc cancels; the ghost is fixed at 180 px screen size while the placed node scales with zoom, so at 0.5x zoom the preview is twice the size of the result.
- **Fix:** Second line "ESC TO CANCEL". Scale the ghost by the current viewport zoom (`reactFlowInstanceRef.current.getViewport().zoom`) via a CSS transform.
- **Files affected:** src/components/GraphCanvas.tsx
- **Implementation notes:** The placement tests from the v1.5 suite assert on the ghost testid, not its size; low risk.
- **Effort:** S

### [Priority: P2] MiniMap category colors and node-count grammar
- **Problem:** MiniMap paints every node amber, wasting the category-color language the canvas already teaches. StatsBar prints "1 NODES". The model name truncates at 240 px even when the bar has space.
- **Fix:** `nodeColor={(n) => ...}` mapping through `opCategoryColor` using the node's data; pluralize; let the name flex with `minWidth: 0` instead of a fixed max.
- **Files affected:** src/components/GraphCanvas.tsx, src/App.tsx
- **Effort:** S

### [Priority: P2] One-prop render culling for big graphs
- **Problem:** React Flow renders every node regardless of viewport; on thousand-node models panning stutters.
- **Fix:** Add `onlyRenderVisibleElements` to the ReactFlow element.
- **Files affected:** src/components/GraphCanvas.tsx
- **Implementation notes:** Verify the jsdom suites still find nodes they assert on (fitView shows all nodes in small fixtures, so they remain rendered); if a test breaks, gate the prop on node count > 300.
- **Effort:** S

### [Priority: P2] Consolidate the styling system
- **Problem:** theme.css and index.css both override .react-flow__controls (theme.css with !important); components hardcode #C0392B, #52C57A, #E8EAF0, #8A8F9E, #5A6070 instead of the tokens that exist for exactly these values. The system is right; its application drifted.
- **Fix:** Delete the theme.css override block in favor of the index.css one; sweep component hex literals to var(--...) equivalents; add the two or three missing tokens (accent-blue #3498DB, the trace green) while at it.
- **Files affected:** src/styles/theme.css, src/index.css, GraphCanvas.tsx, LayerInspector.tsx, App.tsx
- **Implementation notes:** Mechanical; no test risk (tests do not assert computed colors).
- **Effort:** S

### [Priority: P2] Free-text Add Node input count
- **Problem:** The free-text path in the Add Node picker hardcodes inputCount 1 (`commitAddNode(addNodeQuery, 1)`), so a custom two-input op cannot be created by name.
- **Fix:** A small "INPUTS: 1 [-] [+]" stepper row under the text input, passed through to commitAddNode.
- **Files affected:** src/App.tsx (StatsBar picker)
- **Effort:** S

---

## Medium Effort (multi-day)

### [Priority: P0] Decouple graph rendering from WASM session creation
- **Problem:** onnxWorker.ts posts MODEL_LOADED only after `ort.InferenceSession.create` resolves, so first paint of the graph waits on a ~20 MB WASM download plus compile. Verified live: the app sat on "LOADING WASM RUNTIME" for about 60 seconds while the parsed graph was already in hand. First-time visitors will assume it is broken and leave. This is the highest-impact change in this document.
- **Fix:** Post MODEL_LOADED immediately after parsing (both formats). Create the session lazily on the first BENCHMARK command, from the already-retained `exportBuffer`. Benchmark button shows "LOADING RUNTIME" on first use.
- **Files affected:** src/workers/onnxWorker.ts, src/hooks/useOnnxWorker.ts, src/App.tsx
- **Implementation notes:** `exportBuffer.slice(0)` feeds session creation, so the transfer dance is unchanged. Side effect: a structurally-parseable but ORT-invalid model now renders instead of erroring, which is correct viewer behavior (Netron does the same); its invalidity surfaces on benchmark. Progress reporting becomes honest (parse-only) instead of the current fixed 10/50/100. Tests asserting the "Loading WASM runtime" progress stage or MODEL_LOADED ordering need updating; the app suite mocks the worker, so churn is contained to worker-focused tests.
- **Effort:** M

### [Priority: P0] Stop operation errors from destroying the workspace
- **Problem:** Any worker ERROR (a failed benchmark or export, not just a failed load) sets status to `error`, which makes `isReady` false, which mounts the full-screen dropzone over the loaded graph and all pending edits. The user's session appears wiped by a benchmark hiccup. (Nielsen: user control and freedom.)
- **Fix:** Tag ERROR payloads with scope 'load' | 'operation'. Load errors keep today's behavior; operation errors keep status 'ready', keep the graph mounted, and surface through the status line below.
- **Files affected:** src/workers/onnxWorker.ts, src/hooks/useOnnxWorker.ts, src/App.tsx
- **Implementation notes:** The hook's promise-rejection plumbing already distinguishes the pending call; only the status transition needs the scope. Update error-path tests in the hook/app suites.
- **Effort:** M

### [Priority: P0] An annunciator line for silent actions
- **Problem:** The app is pervasively silent: an invalid rewire drop does nothing, bulk delete skips ineligible nodes without saying which or why, COPY gives no confirmation, edge-click inserts a node with no announcement, undo gives no cue of what was undone. (Nielsen: visibility of system status, the biggest cluster of failures found.)
- **Fix:** A single status line, avionics-annunciator style, in the StatsBar (or a 20 px strip under it): dim monospace text that states the last event and auto-clears after ~4 s. Examples: "REWIRE REJECTED: WOULD CREATE CYCLE", "DELETED Conv_12 (CTRL+Z TO UNDO)", "2 OF 5 NODES SKIPPED: AMBIGUOUS RECONNECT", "COPIED". Amber text for rejections, default dim for confirmations. No floating toasts; they would fight the aesthetic.
- **Files affected:** src/App.tsx (an `announce(message, tone)` helper feeding one state value; call sites in handleRewire, applyBulkDelete, handleCopy, undo handler, edge-click path), StatsBar
- **Implementation notes:** `aria-live="polite"` on the line gives screen-reader announcements for free. validateRewire already returns a reason string; today it is thrown away. Add tests for the reject paths.
- **Effort:** M

### [Priority: P1] Accept a dropped model anytime
- **Problem:** After a model loads there is no file input on the page (verified live) and no drop target; replacing the model requires finding "Load new". Every comparable tool (Netron included) accepts a drop at any time.
- **Fix:** Move dragover/drop handling to the app root while a graph is shown; on dragenter, show a full-viewport overlay ("DROP TO REPLACE CURRENT MODEL") reusing the dropzone's visual language; unsaved-edit warning line inside the overlay if edits exist.
- **Files affected:** src/App.tsx, reuse styles from ModelDropzone.tsx
- **Implementation notes:** Guard against dragenter/leave flicker with a depth counter. One new app test.
- **Effort:** M

### [Priority: P1] Motion system: CSS only, glide not bounce
- **Problem:** Structural edits teleport. Deleting a node makes it vanish and every surviving node snaps to its new dagre position in one frame; insert-passthrough and layout toggle do the same. The user cannot visually track what changed, which matters in an editor whose whole job is structural change.
- **Fix:** Four CSS-only pieces, all under the existing 140 to 180 ms / ease idiom:
  1. Re-layout glide: `.react-flow__node { transition: transform 180ms cubic-bezier(0.2, 0, 0, 1); }` and `.react-flow__node.dragging { transition: none; }`. React Flow keys DOM nodes by id, so survivors animate between dagre layouts automatically. This one rule makes delete, insert, rewire, and TB/LR toggle all legible, and it makes exit animations unnecessary: watching neighbors close the gap reads as the deletion.
  2. Node entry: a `node-enter` keyframe (opacity 0, scale 0.97 to 1, 160 ms ease-out) on the OperatorNode inner div, keyed so it fires on mount (new custom/passthrough nodes).
  3. Placement and connection feedback: animate `stroke-dashoffset` on the connection line and on candidate-target handles while dragging (`.react-flow__handle.connectingto` gets an amber pulse via border, not scale).
  4. Loading: the progress bar gets a subtle indeterminate shimmer between stage updates so stalls read as alive.
- **Files affected:** src/index.css (all four), src/components/GraphCanvas.tsx (entry class only)
- **Implementation notes:** Recommendation is explicitly NOT framer-motion or react-spring. Framer Motion is roughly 32 kb gzipped (about 6 kb with LazyMotion but then no AnimatePresence, which is the only feature CSS lacks here), react-spring similar; their sole unique capability (exit animations) is designed around above. This app's baseline is already heavy (onnxruntime WASM), which is exactly why the JS bundle should stay lean; CSS transitions also match the codebase idiom and add zero test risk since jsdom ignores them. Wrap all of it in `@media (prefers-reduced-motion: reduce) { transition: none; animation: none; }`.
- **Effort:** M

### [Priority: P1] Layer Inspector information architecture
- **Problem:** Verified duplication: "Input shapes" renders input tensor names with shapes, then "Inputs" renders the same names again without shapes (same for outputs), so a node's panel says everything twice. The destructive DELETE NODE row sits above the attributes a user came to edit. SENSITIVITY is an unexplained param-count bucket dressed as analysis (an HN commenter will call this out). EXCLUDED YES/NO is ambiguous about whether it shows state or action, and "exclude" itself is never defined.
- **Fix:** (1) Merge into single "Inputs" / "Outputs" sections: one row per tensor, name left, shape right; drop the shape-only sections. (2) Order: identity (op, name), attributes (the editing payload), inputs/outputs, stats (params, size, INT8), then a separated action zone at the bottom (exclude, delete). (3) Rename SENSITIVITY to "QUANT SIZE CLASS" or cut it; if kept, add a title attribute stating it is a parameter-count heuristic. (4) EXCLUDED becomes a labeled action button ("EXCLUDE FROM STATS" / "INCLUDE IN STATS") with a title explaining it only affects rollups and dimming, never export.
- **Files affected:** src/components/LayerInspector.tsx
- **Implementation notes:** Several suites assert on inspector rows (v0.x and app tests reference labels like "Input shapes"); expect a day of test-string updates on top of the component work.
- **Effort:** M

### [Priority: P1] Edge insert needs intent
- **Problem:** Plain click on any edge immediately inserts an Identity node (verified live: node count 3 to 4, silently). Accidental clicks while panning mutate the model. It is also invisible as a feature: nothing anywhere says edges are clickable. Worst of both worlds: dangerous for those who do not know, undiscoverable for those who would want it.
- **Fix:** Clicking an edge selects it (amber highlight) and shows a one-row popover anchored near the click: "INSERT PASSTHROUGH" button plus the tensor name. Escape or clicking elsewhere dismisses. The annunciator line announces the insert with an undo hint.
- **Files affected:** src/components/GraphCanvas.tsx (edge selection state, popover), src/App.tsx (handleEdgeClick becomes two-step)
- **Implementation notes:** v1.2-era tests drive insert-passthrough through edge click; they gain one popover-confirm step. Keep the popover a plain positioned div in the tooltip's existing style.
- **Effort:** M

### [Priority: P1] Stats bar overflow strategy
- **Problem:** Verified at 1024 px: the model name disappears off the left edge and Export Modified / Load new are cropped off the right, with no wrap or scroll. Anyone on a split-screen laptop loses primary actions with no indication they exist.
- **Fix:** Priority collapse: below ~1280 px drop stat labels ("2.4M" instead of "2.4M PARAMS", title attributes carry the words); below ~1100 px collapse Benchmark/Download/Load-new into a single "..." menu, keeping Export Modified (the primary action) and the filter always visible.
- **Files affected:** src/App.tsx (StatsBar), src/index.css (container query or a matchMedia hook)
- **Implementation notes:** CSS container queries handle this without JS if StatsBar moves its inline styles to classes, which the tokenization sweep above sets up. The "..." menu can reuse the Add Node dropdown pattern.
- **Effort:** M

### [Priority: P1] Keyboard and ARIA pass on the editing surface
- **Problem:** Attribute editing is mouse-only: values are divs with onClick, unreachable by Tab, so keyboard users cannot use the product's headline feature at all. Delete-picker options are also divs. The filter dropdown has listbox/option roles but the input lacks combobox wiring (aria-expanded, aria-activedescendant), and nodes have no focus outline in canvas.
- **Fix:** Attribute value cells and delete-picker options become buttons (styled identically; the reset in index.css already normalizes button chrome), Enter starts editing, existing Esc/Enter handling stays. Add combobox ARIA to the filter input. Add `.react-flow__node:focus-visible { outline: 1px solid var(--color-amber); }` and set node aria-labels ("Conv, 2.3M params").
- **Files affected:** src/components/LayerInspector.tsx, src/App.tsx (StatsBar), src/index.css, src/components/GraphCanvas.tsx
- **Implementation notes:** Tests select attr cells by data-testid, which survives the div-to-button swap; a few fireEvent.click calls may need to become button-role queries. React Flow v12 already gives nodes keyboard focus and arrow-key movement; the outline makes it visible.
- **Effort:** M

### [Priority: P2] Box select
- **Problem:** Multi-select exists only via ctrl-click accumulation; selecting twenty nodes for a bulk exclude/delete is tedious. React Flow ships marquee selection; it is currently unused.
- **Fix:** Enable `selectionOnDrag` with `panOnDrag={[1, 2]}` (left-drag selects, middle/right-drag pans) or shift-drag selection, plus `onSelectionChange` syncing into the existing applySelection path.
- **Files affected:** src/components/GraphCanvas.tsx, src/App.tsx
- **Implementation notes:** The selection-sync loop needs care to not fight setMultiSelection (guard on set equality). Add a test for selection-change wiring.
- **Effort:** M

---

## Larger Initiatives (needs a design decision from you first)

### [Priority: P0] The exported-model validity story
- **Problem:** Rewire validation is structural only (cycles, self-connection); nothing checks arity, shape, or dtype compatibility, so Forma will happily export a model that no runtime can load, without a word of warning. For an HN launch whose pitch is "edit and re-export, it just works", the first top comment will be someone pasting the onnxruntime stack trace from a Forma export. The roadmap correctly parks full shape inference in the backlog; the question is what ships before the public launch.
- **Open question:** Which tier do you want for v2.0? (a) Disclaimer only: an "unvalidated edit" indicator on structurally-edited exports plus a line in the export flow; near-zero effort, honest, but concedes the point. (b) Arity and dtype checks: validate input counts per op (a small table for the ~40 common ops) and tensor elem-type agreement on rewire; catches most foot-guns, no shape inference; roughly a week. (c) A post-edit "verify" button that round-trips the exported bytes through `ort.InferenceSession.create` in the worker and reports pass/fail; this reuses infrastructure you already have and turns the weakness into a demo moment ("Forma verifies the export loads"). My recommendation: (c) plus the disclaimer, and skip (b) until users hit real cases; session-create validation is stronger than any static table you will write.
- **Files affected:** onnxWorker.ts, App.tsx, StatsBar; graphUtils.ts if (b)
- **Effort:** L (option c alone is M)

### [Priority: P1] Big-model strategy
- **Problem:** dagre runs synchronously on the main thread inside a useMemo, and the whole graph re-lays on every structural edit. At ResNet-50 scale (~125 nodes) this is fine; at BERT (~1300 nodes) or an SD UNet (~2000) the tab will freeze for seconds per edit, and React Flow will crawl even with culling. Someone on HN will drop a 1 GB LLM export in the first hour, and "browser tool freezes" is the comment you cannot delete.
- **Open question:** What is the supported ceiling, and what happens above it? Options: (a) hard honesty: above N nodes show "graph too large for interactive layout, showing first N by size" with a filter escape hatch; cheapest. (b) Move dagre into the worker with a layout-pending shimmer; keeps full graphs but edits still relayout globally. (c) Netron-style block collapsing (group repeated subgraph patterns); the real answer and a genuine differentiator, but it is a multi-week feature that touches parsing, layout, and interaction. Recommendation: (b) plus a soft warning banner above ~800 nodes now; treat (c) as a post-launch headline feature.
- **Files affected:** GraphCanvas.tsx, onnxWorker.ts or a second layout worker, App.tsx
- **Effort:** L

### [Priority: P1] Desktop-only gate versus mobile support
- **Problem:** Verified at 390 px: the fixed 280 px inspector consumes most of the viewport, the stats bar is cropped to a fragment, and the core interactions (drag-to-rewire onto 6 px handles, ctrl-click, hover pencils) are mouse-dependent. Mobile is not degraded; it is unusable. Meanwhile roughly half of HN reads on a phone, and their first impression is this broken state.
- **Open question:** Gate or invest? (a) Ship a sub-900 px screen: wordmark, one-liner, "Forma is a desktop tool: open on a larger screen", GitHub link, and a static product screenshot so phone visitors still see what it is; half a day, and honestly all a launch needs. (b) A read-only mobile viewer (pan/zoom canvas, tap to inspect via bottom sheet, all editing hidden); roughly two weeks and it duplicates inspector layout work. Recommendation: (a) now, revisit (b) only if mobile analytics justify it post-launch. The screenshot in the gate is the important part; it converts phone visitors into desktop return visits.
- **Files affected:** App.tsx (viewport gate), one static asset
- **Effort:** S for (a), L for (b)

### [Priority: P2] Command palette
- **Problem:** The keyboard story is fragmented: `/` reaches node search, but layout, add-node, benchmark, export, and exclude have no keyboard path, and each new feature has been adding another one-off StatsBar control. Dev-tool users expect Ctrl+K.
- **Open question:** Is the palette the unifying interaction, with the filter input becoming just one of its modes? Committing means new features land as palette commands first (cheap, discoverable, keyboard-native) and the StatsBar stops accreting buttons; not committing means investing in the individual controls above instead. The two paths overlap heavily in effort. If v2.0's shareable links land, "copy share link" is exactly the kind of action a palette makes discoverable without UI chrome.
- **Files affected:** New component, App.tsx keydown routing; pairs well with the annunciator line
- **Implementation notes:** Build it in-house (a filtered list over a static command array, in the search dropdown's existing style); do not add cmdk or kbar, which drag in styling assumptions the design system will fight.
- **Effort:** L

### [Priority: P2] How much onboarding is too much for this aesthetic
- **Problem:** The austerity is the brand, but right now zero affordances means the median visitor never learns that attributes are editable (hover-only pencil), that edges rewire by dragging handles, or that nodes delete. The features that make Forma unique are the least discoverable things in it.
- **Open question:** Pick a guidance level. (a) Passive only: the shortcuts overlay, title attributes everywhere, and the annunciator line teaching through feedback; zero visual noise, teaches only on interaction. (b) One-time hint chips: on first model load, three dismissible dim-amber labels anchored to a node, an edge, and the inspector ("drag handles to rewire", "click values to edit"), never shown again via localStorage. (c) A scripted first-run tour: rejected out of hand, it would poison the terminal aesthetic. Recommendation: (a) shipped in the items above plus (b) behind a decision from you; (b) is the only way a first-time HN visitor learns drag-to-rewire exists during their 60-second visit, and dismissed-forever chips are a defensible compromise.
- **Files affected:** GraphCanvas.tsx, App.tsx, localStorage flag
- **Effort:** M

---

## Suggested sequencing for a launch

1. Week 1, the door: sample model, empty-state identity, title/OG tags, self-hosted font, error message, dead-dependency cleanup. All S items; the product stops losing visitors it never got to impress.
2. Week 2, the first minute: decoupled WASM session (P0), annunciator line, edge-insert intent, contrast fixes, shortcuts discoverability, benchmark warmup.
3. Week 3, the workspace: error-scope split, motion system, inspector IA, stats-bar collapse, drop-anytime, keyboard/ARIA pass.
4. Then the three launch-gating decisions: validity story (recommend option c), big-model ceiling (recommend worker layout plus banner), mobile gate (recommend the gate). None of these should slide past the v2.0 HN post.
