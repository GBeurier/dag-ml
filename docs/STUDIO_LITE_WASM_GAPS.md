# dag-ml — gaps for the in-browser studio-lite (WASM execution)

`nirs4all-lite/studio-lite` drives **dag-ml-wasm** in the browser: it compiles the pipeline
DSL → GraphSpec, builds the execution plan, runs FIT_CV through `execute_campaign_phase_json`
with a synchronous JS controller (libn4m numerics), and selects with
`select_candidates_json`. The DSL surface is rich and **compiles fully** (verified with
`dag-ml-cli compile-pipeline-dsl`: Branch / Merge / MergeModel / ConcatTransform,
GeneratorStep {Or, Cartesian}, param_generators). The gaps below are in **browser
EXECUTION**, which currently forces host-side (TypeScript + libn4m) workarounds. Closing them
lets the real dag-ml scheduler run the whole graph with lineage/replay.

This is a forward-development backlog, not a bug list. None of it weakens OOF/leakage/fold
safety — the workarounds preserve those; the goal is to move them into the core.

## 1. Per-node provider execution for multi-node graphs  *(the load-bearing one)*
`execute_campaign_phase_json` reliably executes only **model-only** graphs through the JS
controller. A multi-node graph (a preprocessing chain, a branch, a concat) trips
`planning_failed: no controller registered for node 'transform:compat.N'` because the browser
execution path has **no per-node data provider** wiring. studio-lite works around this by
running the leakage-honest *preprocessing + model* chain on libn4m over dag-ml's `FoldSet`
(TS orchestration in `orchestrate.ts`) — i.e. **not** through the dag-ml scheduler.

To run preprocessing / branch / merge graphs through dag-ml, the browser needs a
provider-per-node execution path: `dag-ml-data`'s `WasmInMemoryProvider` feeding each node's
input/output views so the scheduler can drive transform nodes with data flowing node→node
(handles across the boundary, identity-keyed). This is the single biggest unlock.

## 2. Branch / Merge / MergeModel / ConcatTransform execution
The DSL lowers these correctly (compiles to `feature_join` nodes), but the browser executes
only a **feature-union** (TS column-concat of branch sub-chains on libn4m). Real branch
fan-out, merge fan-in, and model-stacking (`MergeModel`) execution need #1 (per-node
provider) plus controller handling of branch/merge by sample identity.

## 3. Generator execution — Cartesian + zip/chain/sample
**OR** generators execute today (the host reads `ExecutionPlan.variants`, runs per-variant
FIT_CV, and calls `select_candidates_json`). **Cartesian** *compiles* but the browser does
not expand the cross-product (it shows a guard); the studio editor also offers
zip/chain/sample generator kinds. Either:
- (a) expose a wasm `enumerate_variants_json(plan)` returning the full variant list (incl.
  cartesian/zip/chain/sample) so the host runs each; or
- (b) drive all-variant FIT_CV inside `execute_campaign_phase_json` and return per-variant
  results.
Today the host only re-derives OR variants.

## 4. `execute_campaign_phase_json` variant semantics
The scheduler loops **all** `plan.variants` in one execute call with no per-variant pin; the
host does ONE execute and buckets OOF by `lineage.variant_id` (works, but it's coupling). A
documented "one-execute-all-variants" contract — or a `variant_id` pin so REFIT/PREDICT can
target the selected variant — would harden it. REFIT of the winning variant is currently
pinned host-side.

## 5. Classification metrics + SELECT scoring in the core
`metrics.rs` exposes regression metrics only. The browser computes classification metrics
(accuracy / F1 / confusion) in TS for display and builds `CandidateScore` host-side. Moving
classification metrics into the core + a `score_predictions_json` (regression + classification)
would make selection scoring fully dag-ml-owned and consistent with lineage.

## 6. REFIT / PREDICT phases through dag-ml
The browser runs REFIT (full-train fit) and PREDICT directly with libn4m, not through
dag-ml's REFIT/PREDICT phases. Wiring those phases to the JS controller (with #1) would make
the entire COMPILE→PLAN→FIT_CV→SELECT→REFIT→PREDICT loop dag-ml-driven, giving lineage and
replay for prediction too.

---
*Source: the nirs4all-lite/studio-lite build (2026-06). dag-ml-wasm already exposes
compile / plan / fold-build / execute(FIT_CV) / select; the gaps are multi-node + branch +
non-OR-generator execution, core-side metrics, and the REFIT/PREDICT wiring.*
