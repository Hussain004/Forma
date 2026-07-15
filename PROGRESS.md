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
- [ ] P0 Humanize the malformed-file error
- [ ] P1 Self-host JetBrains Mono
- [ ] P1 Remove the dead reactflow 11 dependency and starter leftovers
- [ ] P1 Rename "Download" and fix its filename
- [ ] P1 Fix dim-text contrast and micro type sizes
- [ ] P1 Make shortcuts discoverable
- [ ] P1 Benchmark warmup and running state
- [ ] P2 Layout toggle that says what it does
- [ ] P2 Placement-mode polish
- [ ] P2 MiniMap category colors and node-count grammar
- [ ] P2 One-prop render culling for big graphs
- [ ] P2 Consolidate the styling system
- [ ] P2 Free-text Add Node input count

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

- Real title + OG/Twitter meta tags in index.html; generated public/og-image.png (a static Avionics-Blueprint-styled card, not a live screenshot, since no sample model existed yet at this point in the work).
