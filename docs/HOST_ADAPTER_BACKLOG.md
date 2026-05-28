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
| 2 | **prospectr (R)** ✅ shipped | R | JSONL | 2 (delivered) | scaffold for #3 | Shipped through commits G.1–G.2: `examples/adapters/prospectr_process_controller.R` builds the R-side JSONL scaffold from scratch (jsonlite-backed describe fast path, structured `AdapterTaskError` condition, fold/REFIT/PREDICT partition leakage checks, lifecycle markers) and dispatches the stateless prospectr operators `SNV`/`standardNormalVariate`, `savitzkyGolay`, `gapDer`, `binning`, `continuumRemoval`; `examples/controllers/prospectr.controller.json` declares the matching transform-kind ControllerManifest with the same alias-set parity test pattern as F.3. `msc` is excluded — its reference spectrum is fitted on the calibration set and applying the batch's own `colMeans` at predict time would leak validation data, so MSC needs the stateful artifact path tracked separately. |
| 3 | **mdatools (R)** | R | JSONL | 2–3 (revised down) | full reuse of #2 scaffold | The R-side JSONL loop, `AdapterTaskError` handling, lifecycle markers, leakage checks, and synthetic feature smoke from G.1 are reusable. mdatools operators (`pls`, `pcr`, `simca`, `mcr.als`, `pca`) are stateful so the new cost is the RData-based artifact persistence (mirror of sklearn's joblib path) and per-operator fit/predict wrappers. Cross-validation is owned by `dag-ml`, mdatools fits a fold at a time. |
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

Items #1 (sklearn production) and #2 (R prospectr) are shipped.
The next slice is **#3 (R mdatools)** which now reuses the R-side
JSONL scaffold built in G.1 and the sklearn-side artifact-persistence
pattern from F.1, leaving the new work focused on RData-backed
artifact storage and per-operator fit/predict wrappers. After #3,
the path follows the SpectroChemPy / Orange-Spectroscopy Python
adapters (#4 and #5) which reuse the sklearn production scaffold,
and finally the YAML controller registry (#6). A separate slice
should add stateful MSC handling on the prospectr controller (excluded
from G.1–G.2 to keep the prospectr controller honestly stateless).
