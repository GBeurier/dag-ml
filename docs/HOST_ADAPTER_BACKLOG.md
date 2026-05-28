# Host Adapter Backlog

This document tracks the honest production status of host controller
adapters: what is shipped, what is missing, what the wire protocol
actually is, and how much each remaining adapter would realistically
cost. It is the source of truth for the "production host controller
adapters with native libraries" item that has lived in
`docs/STATUS.md`'s "Not implemented yet" list for several releases.

## Wire protocol: process adapter JSONL

The C ABI controller vtable (`DagMlControllerVTable`) is an
in-process abstraction for Rust controllers compiled into the same
binary as the scheduler. **It is not the path for Python/R hosts.**

The only stable wire protocol for cross-language host controllers is
the **process adapter JSONL contract**, defined by:

- `docs/contracts/process_adapter_description.schema.json` v1 —
  the handshake (`adapter_id`, `supported_modes`, capabilities).
- `docs/contracts/process_adapter_frame.schema.json` v1 — the
  framing: `init` → `ack`, `task` → `result | error`, `close` → `ack`.
- `docs/contracts/node_task.schema.json` v1 + `node_result.schema.json`
  v1 — payload shapes.

Required capabilities for any production adapter:
`control_frames_v1`, `node_task_json_v1`, `node_result_json_v1`,
`parallel_invocation_v1`. Recommended:
`persistent_workers`, `worker_env`, `stateful_refit_artifacts`.

The reference is `examples/adapters/sklearn_process_controller.py`
(~540 LOC). Lines 16–25 enumerate the capabilities. The `--describe`
handshake fires from an early fast-path guard at lines 76–78, before
`import numpy` at line 81 — keeping the coordinator's discovery call
cheap and sklearn-free. The runtime `--jsonl` loop lives in `main()`
at lines 527–542. The existing JSONL loop, lifecycle markers,
stateful model cache, and fake-data passthrough are all reusable
scaffolding.

## What's already shipped

| Layer | Status |
|---|---|
| Persistent process pool, init/task/close framing, retry/restart on flaky workers, env vars | **shipped** in `crates/dag-ml-cli/src/main.rs` (lines ~3040–3300) |
| `--describe` handshake validation + capability gating | **shipped** in core |
| Process adapter description + frame JSON Schemas | **shipped** in `docs/contracts/` |
| sklearn smoke adapter (Ridge + StandardScaler, JSONL, stateful) | **shipped** as `examples/adapters/sklearn_process_controller.py` |
| Generic Python adapter scaffold | **shipped** as `examples/adapters/python_process_controller.py` |
| Flaky adapter for retry coverage | **shipped** as `examples/adapters/flaky_process_controller.py` |

So the **infrastructure is complete**. What remains is real adapters
covering production operator catalogs and a YAML registry to declare
them.

## Backlog

Slice = a focused, gated, Codex-reviewed chunk of work (≈1 commit per
slice in the style used throughout the recent phases).

| # | Adapter | Language | Wire | Slices | Existing scaffold | Notes |
|---|---|---|---|---|---|---|
| 1 | **sklearn (production)** ✅ shipped | Python | JSONL | 3 (delivered) | promoted from smoke adapter | Shipped through commits F.1–F.3: `examples/adapters/sklearn_production_controller.py` extends `operator_selectors` to cover sklearn.preprocessing/linear_model/ensemble/decomposition (24 classes) with `joblib.dump`/`joblib.load` artifact persistence under `$DAG_ML_PROCESS_ARTIFACT_DIR` (basename-confined; absolute and parent-traversal URIs rejected); structured `AdapterTaskError` frames keep the persistent worker alive across bad tasks; `signal.SIGALRM`-based fit timeout from `$DAG_ML_PROCESS_FIT_TIMEOUT_SECONDS` surfaces as a retryable `fit_timeout`; `examples/controllers/sklearn_production.controller.json` declares the matching `ControllerManifest`, with a Rust test that asserts the manifest's `aliases` selector matches the controller's runtime `OPERATOR_SELECTORS` registry exactly. |
| 2 | **prospectr (R)** | R | JSONL | 3–5 | none | R package `dagml.controller.prospectr`. JSONL loop in R (no native bindings needed) requires a from-scratch R-side stdin/stdout reader on top of `jsonlite` and a `data.frame ↔ matrix` adapter at the boundary — no R scaffolding exists in the workspace. Dispatch to prospectr's `standardNormalVariate`, `msc`, `savitzkyGolay`, `gapDer`, `binning`, `continuumRemoval`. Translate `NodeTask.data_views` Arrow handles to R `data.frame`/`matrix` on entry. Higher slice ceiling than Python adapters reflects this lack of reusable scaffolding. |
| 3 | **mdatools (R)** | R | JSONL | 3–5 | partial reuse of #2 | Same package shape as prospectr; the R-side JSONL loop and `data.frame ↔ matrix` adapter built for #2 are reused, but the operator surface is more complex (`pls`, `pcr`, `simca`, `mcr.als`, `pca`) and each operator's R-side fit/predict signature must be wrapped individually. Cross-validation is owned by `dag-ml`, mdatools fits a fold at a time. |
| 4 | **SpectroChemPy (Python)** | Python | JSONL | 2 | sklearn pattern reuse | **Python, not R** despite occasional grouping with R libs. Pattern reuses sklearn adapter scaffold. Operators from `spectrochempy.analysis.*` and `spectrochempy.processing.*`. NMR/IR-specific operators benefit from `AxisKind::Wavenumber` shipped in Phase D. |
| 5 | **Orange-Spectroscopy (Python)** | Python | JSONL | 2 | sklearn pattern reuse | **Python, not R**. Add-on for Orange Data Mining (`orangecontrib.spectroscopy`). Operators: preprocess (SNV, MSC, baseline), models (Stagewise, IntegrateSimps). Smaller community than mdatools/prospectr; lower priority. |
| 6 | **ControllerManifest YAML registry** | Rust (CLI) | n/a | 1 | none | Declarative YAML (`controllers/<adapter>.controller.yaml`) for the 5 adapters above. Each declares `controller_id`, `version`, `operator_kind`, `operator_selectors`, `capabilities`, `fit_scope`, `process_adapter`. Validated at registry load through the existing `ControllerManifest::validate`. |

## Cost estimate

≈ **13–18 slices total** (sum of the table above; R adapters carry
the wider 3–5 ceiling because no R-side process-adapter scaffold
exists in the workspace). The first ≈70% of the sklearn adapter
already exists as the smoke fixture — what's missing is expanded
`operator_selectors`, `joblib.dump` artifact persistence, a declared
`ControllerManifest`, and a structured error model. Promoting that
smoke into a production controller is the cheapest item and unblocks
`nirs4all` end-to-end. R adapters are individually more expensive
than the Python ones because every R adapter has to build its own
`stdin`/`stdout` JSONL loop, its `data.frame ↔ matrix` boundary
adapter, and its `jsonlite` framing — none of which exist in the
workspace today. Item #2 (prospectr) carries the up-front R-side
scaffolding cost; item #3 (mdatools) reuses it.

## Out of scope (explicitly)

| Item | Why |
|---|---|
| Native bindings (PyO3, libR-sys, JNI) for any of these adapters | The JSONL protocol is the stable contract. Native bindings duplicate effort, double the maintenance burden, and do not improve performance for non-GIL-bound operators. |
| C ABI controller vtable (`DagMlControllerVTable`) wrappers in Python/R | The C ABI is for in-process Rust controllers only. Python/R use process adapter JSONL. |
| MATLAB / Julia / Octave controllers | Out of scope for the current release. Could be added later through the same JSONL contract. |
| GPU-aware host controllers | Out of scope; the controller batching contract supports this in principle (`accepts_task_batch` capability) but no production GPU adapter is on the immediate roadmap. |

## Naming sanity

The original ask grouped "prospectr, mdatools, SpectroChemPy,
Orange-Spectroscopy" as **R libs**. That is correct for prospectr and
mdatools but **not** for SpectroChemPy (Python) or Orange-Spectroscopy
(Python add-on for Orange Data Mining). This backlog separates them
honestly: 2 R adapters, 2 additional Python adapters, plus the
production sklearn adapter.

## Next slice

Item #1 (sklearn production) is shipped. The next slice is
**#2 (R prospectr)**: it carries the up-front R-side JSONL
scaffolding cost (stdin/stdout reader on top of `jsonlite`,
`data.frame ↔ matrix` boundary adapter, lifecycle marker support).
Item #3 (R mdatools) then reuses that scaffold and is cheaper. Both
unblock R-based NIRS workflows after item #1 unblocked the Python
sklearn path.
