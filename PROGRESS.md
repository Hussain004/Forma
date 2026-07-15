# IMPROVEMENTS.md progress tracker

Tracks work against IMPROVEMENTS.md, one line per item. Updated as items land.
Branch: `ux-improvements`. Larger Initiatives are intentionally skipped -- each one
has an open question that needs an answer from the project owner before code should
be written (see IMPROVEMENTS.md's "Larger Initiatives" section for the options).

Legend: [ ] not started, [~] in progress, [x] done, [-] skipped (needs a decision)

## Quick Wins

- [x] P0 Real title and social meta tags
- [ ] P0 Bundle a sample model
- [ ] P0 Product identity on the empty state
- [x] P0 Humanize the malformed-file error
- [x] P1 Self-host JetBrains Mono
- [x] P1 Remove the dead reactflow 11 dependency and starter leftovers
- [x] P1 Rename "Download" and fix its filename
- [x] P1 Fix dim-text contrast and micro type sizes
- [ ] P1 Make shortcuts discoverable
- [ ] P1 Benchmark warmup and running state
- [x] P2 Layout toggle that says what it does
- [x] P2 Placement-mode polish
- [x] P2 MiniMap category colors and node-count grammar (model-name flex-width deferred to the stats bar responsive collapse item, where the bar's overflow behavior gets rebuilt properly)
- [x] P2 One-prop render culling for big graphs (gated on `onnxNodes.length > 300`, not always-on -- jsdom reports zero-size bounding rects in tests, which would cull every node in every small fixture if this ran unconditionally)
- [ ] P2 Consolidate the styling system
- [x] P2 Free-text Add Node input count

## Medium Effort

- [ ] P0 Decouple graph rendering from WASM session creation
- [ ] P0 Stop operation errors from destroying the workspace
- [ ] P0 An annunciator line for silent actions
- [ ] P1 Accept a dropped model anytime
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

- Vendored the JetBrains Mono latin subset (weights 300/400/500/700) into public/fonts/ and replaced the Google Fonts @import with local @font-face rules. Sourced the files from the @fontsource/jetbrains-mono npm package (proper licensing, no manual font wrangling), then uninstalled the package itself since only the static files were needed. Verified live: zero external hosts contacted on load (previously fonts.googleapis.com and fonts.gstatic.com), which makes the "your model never leaves the browser" pitch literally true.

- Malformed-file error now shows a short friendly headline ("This file doesn't look like a valid ONNX or TFLite model.") with the raw onnxruntime message demoted to a smaller, non-shouting sentence-case line underneath, instead of dumping the full ALL-CAPS ORT error as the primary message. Verified live against a real junk file.

- "Download" (which returns unmodified original bytes) renamed to "Download Original" with a clarifying title, and its filename suffix fixed from the misleading `_export.onnx` to `_original.onnx`. "Export Modified" now shows a live edit count, e.g. "Export Modified (3)", from `attrOverrides.size + structuralOps.length`.

- Raised `--text-dim` from #4A4F5E (~2.2:1, WCAG fail) to #7A8191 (~4.5:1, WCAG AA pass) and pointed the hardcoded #5A6070 shape-label color and #3A4050 idle-pencil color at the same token instead of their own separate low-contrast values. Bumped MOD/NEW node badges from 7px to 9px and shape labels from 9px to 10px. Verified live: node shape labels ([1, 4]) went from barely visible to clearly legible.

- Placement ghost now scales with canvas zoom (via `onViewportChange`, CSS transform: scale from center) instead of staying a fixed screen size while the real node scales with the viewport; added an "ESC TO CANCEL" hint. Verified live at a zoomed-out viewport (screenshot: ghost visibly smaller than the real node behind it).

- Free-text Add Node entry gets an input-count stepper (1-8, defaulting to 1) instead of hardcoding a single input; stepper buttons use preventDefault on mousedown so they don't blur the text field and close the dropdown. New test + verified live.

- Layout toggle now reads "LAYOUT TB" / "LAYOUT LR" with a title attribute; MiniMap nodes colored by op category instead of uniform amber; node count singularizes to "1 NODE"; ReactFlow gets `onlyRenderVisibleElements` gated behind a 300-node threshold. Verified live in a production preview build (screenshot: minimap shows the Relu node green, matching its Activation category).

- Removed the unused `reactflow` v11 package (nothing imports it, `@xyflow/react` v12 is the real dependency) and Vite-starter leftovers (react.svg, vite.svg, empty App.css). Left hero.png alone since it isn't referenced anywhere and wasn't called out in the audit -- may be intentional for future README use.

- Real title + OG/Twitter meta tags in index.html; generated public/og-image.png (a static Avionics-Blueprint-styled card, not a live screenshot, since no sample model existed yet at this point in the work).
