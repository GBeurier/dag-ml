# DAG-ML — Summary: objective, technical decisions, roadmap

Entry document. Read this first. For details on the mechanisms, see the
working doc `dag_ml_polyglot_core_design.md`; for the complete ML contract, the
specs `dag_ml_specification_v1.md`, `dag_ml_use_cases.md`,
`ml_data_specification_v1.md`.

Core language: **Rust** (decided). Existing methods: **C++** (nirs4all-methods,
retained). Loaders: **Rust** (nirs4all-io, in progress).

---

## 1. Objective

DAG-ML is a **local, in-process ML engine**, extracted and generalised from nirs4all,
which formalises: compilation of a DSL pipeline into a DAG, enumeration of variants,
multi-phase execution (`COMPILE → PLAN → FIT_CV → SELECT → REFIT → PREDICT →
EXPLAIN`), CV/OOF/stacking **without leakage**, refit, predict/explain by replay,
artifact/cache/lineage stores, parallelism, multitask batching on the controller
side, and a stable operator contract.
No NIRS logic inside: the domain enters through plugins.

The **goal of the polyglot architecture**: to bring the **ML rigour engine**
(graph, OOF, folds, lineage, determinism) to several languages
(Python, R, JS, native) **without rewriting the operator ecosystem**. The aim is,
ultimately, the same engine — hence the same guarantees — usable from Python
(sklearn/torch), R (mlr3), or as a full-native build.

---

## 2. The guiding idea

**The core is a control plane, not a compute kernel.** It does virtually
no FLOPs: heavy computation lives in the operators (sklearn/torch/C++), already
native. The core owns the **graph, the phases, the folds, the OOF, the lineage,
the scheduler, the RNG** — the invariants.

Two principles encoded in the ABI:

1. **Visibility** — the core sees in Arrow exactly three things: the **identity**
   (sample/observation/target/group/origin ids), the **predictions**, and
   **y_true** (scoring). **Never X or the features.** Everything else = **opaque
   handles** (u64) resolved on the host side.
2. **Ownership** — the core owns the **handle lifecycle**; the host
   owns the **underlying object** and frees it on `release`. Arrow carries its
   own release callback.

Violating (1) = we re-marshal the heavy data. Violating (2) = use-after-free cross-FFI.

---

## 3. Architecture

```
RUST                                   C++                         per language
┌──────────────────────────┐          ┌─────────────────┐        ┌────────────────┐
│ dagml-core (control)      │  ABI C   │ nirs4all-methods│  pyo3  │ sklearn/torch  │
│  graph, folds, OOF,       │◄────────►│  (controllers   │◄──────►│ (per language) │
│  lineage, scheduler, RNG  │ extern"C"│   validated C++)│ extendr│ mlr3 (R)       │
│ nirs4all-io (loaders)     │          └─────────────────┘ napi/  │ tfjs/onnx (JS) │
│  ── identity+preds=Arrow ─│                              wasm    └────────────────┘
└──────────────────────────┘
  data    : Arrow C Data Interface (zero-copy)  /  DLPack (tensors, GPU)
  models  : ONNX / safetensors (cross-language inference)
  control : core determinism (folds, OOF, RNG)
```

**Three layers, three language statuses** (true for both data *and* operators):

| Layer | Content | Status |
|---|---|---|
| Metadata / plan | schema, axes, representations, GraphSpec, DataPlan, ModelInputSpec, fingerprint | **neutral** (core) |
| Reasoning | find_path, planning, OOF join, scheduling, control RNG | **neutral** (Rust core) |
| Buffers + execution | DataBlock, fit/predict/transform, collation | **per language** (controllers) |

---

## 4. Technical decisions

| Decision | Justification (1 line) |
|---|---|
| **Core in Rust** | The project's weak point (handle liveness cross-FFI + concurrent scheduler) is Rust's strength; binding/Arrow stack proven (Polars, pydantic-core, tokenizers); consistent with nirs4all-io already in Rust. |
| **Methods in C++ behind the C ABI** | Validated/portable code, zero rewrite; called as controllers via `extern "C"` vtable, zero call overhead. |
| **C ABI between core and ALL controllers** | A single boundary, symmetric Python/R/native; neutralises the need to couple core and methods. |
| **Data via handles, never in the clear** | The buffer stays on the host; only identity + predictions cross (Arrow). Marshalling the heavy data = zero intra-process. |
| **Arrow C Data Interface** for what the core reads | Zero-copy, stable ABI, polyglot lingua franca (py/R/JS). |
| **DLPack** for dense tensors | Zero-copy CPU/GPU (torch/tf/jax). |
| **ONNX/safetensors** for fitted models | Cross-language inference (≠ Arrow, which carries the data). |
| **Counter-based splittable RNG in the core** | Cross-language determinism **and** scheduling-independent for the entire control plane. |
| **pyo3/maturin** as primary binding; extendr (R), wasm-bindgen (JS) next | Mature tooling, wheels without CMake. |
| **Identity-native splitters in the core** (KFold, GroupKFold…) | No data needed → Tier 1 RNG, cross-language identical. Feature-based (KS/SPXY) = controllers. |

---

## 5. Boundary contracts (ABI)

Two C vtables `#[repr(C)]` (full detail: working doc §9-10). The essentials:

**`ControllerVTable`** (a sklearn/torch/mlr3/C++ operator):
- `clone_with(op, params)` — lazy construction of a variant (the core drives *which*
  params, the host *how*).
- `describe(op) → blob` — PLAN contract (ModelInputSpec, ports, phases, flags;
  versioned canonical JSON format).
- `fit / transform_fit` → fitted handle; `predict` → **Arrow predictions**;
  `transform_apply` → data handle; `invert` (y-transform).
- `split` (identity + optional data for KS/SPXY) → Arrow fold table.
- `serialize / deserialize` (joblib/onnx/safetensors), `cache_key`,
  `release / free_bytes / destroy`.
- `capabilities` (bitset): `GIL_FREE_COMPUTE`, `THREAD_SAFE`, `STATEFUL`,
  `INVERTIBLE`, `DETERMINISTIC`, `RNG_FROM_CORE`, `REQUIRES_DATASET_PLAN`,
  `EMITS_RELATION`, `ACCEPTS_TASK_BATCH`, `ACCEPTS_STATIC_SUBGRAPH`…

**`DataVTable`** (the per-language data layer):
- `materialize`, `make_view` (slice **by sample-ids**, never by positions →
  anti-leakage carried by the ABI), `view_identity`, `target_arrow`, `feature_arrow`,
  `ingest_arrow` (OOF → handle), `handle_nbytes`, `schema_fingerprint`,
  `release / destroy`.

**What crosses**: input = handles (u64) + scalars + blobs; output =
handles + Arrow (identity, predictions, relations). The heavy buffer never crosses
intra-process.

---

## 6. Data & dimensions

A dimension is not an integer: it is a **semantic axis** (`AxisSpec{kind,
unit, size, coordinates}`). Model compatibility is a **path search**
(`find_path`, Dijkstra) from the `native_representation` of a source to an
accepted representation of the `ModelInputSpec`, producing a `DataPlan`
(`materialize → adapt* → align → join → collate`). Decided at PLAN on the schema
alone; refusable; escalatable if ambiguous.

- **Block per data type** = `DataTypePlugin` (+ adapters + collator): signal,
  image, genotype, time series, graph, table, text. Unit of extension.
- **Collation = last**: padding/batch/channel order only at the model boundary
  (the Arrow-column ↔ dense tensor impedance is isolated there).
- **Language impact**: the shape algebra (schema, find_path, DataPlan) is
  **neutral** (core); the buffers + adapter execution + collation are
  **per language**. nirs4all-io (Rust) makes **ingestion identical** cross-language.

---

## 7. RNG & reproducibility

**Counter-based splittable** PRNG (Philox/Threefry) in the core; `SeedContext`
= tree of derived streams from the path (`SHA256(path)[:16] → key`). Two tiers:

- **Tier 1 — control randomness** (splits, tuner sampling, augmentation selection):
  owned by the core, **cross-language bit-identical** and scheduling-independent.
  Pre-drawn in Arrow or via upcall.
- **Tier 2 — internal framework randomness** (NN init, sklearn bootstrap): seed
  passed, reproducible **intra-lib** only.

**Cross-language reproducibility** (Python ≡ R) on every shared node, under 5
conditions, all under your control: (1) consistent binding compilation;
(2) deterministic-order reductions; (3) controlled linear algebra (no divergent
system BLAS); (4) control randomness exclusively from the core; (5) same
ingestion (nirs4all-io) + same dtype. Divergence confined to **language-specific
models** (torch/sklearn vs mlr3). Identical cross-language model = shared native
C++ method, or same artifact replayed via ONNX.

---

## 8. Roadmap

Proposed build order, each phase delivering something verifiable.

**Phase 0 — Freeze the contracts.**
- Neutral types `ml_data.contract` (schema, axes, representations, ModelInputSpec,
  DataPlan, FusionPolicy, SampleRelation…) — already specified.
- The C ABI: `ControllerVTable`, `DataVTable`, `describe` blob format, Arrow/handle/ownership
  conventions. *DoD: C header + Rust type crate, versioned.*

**Phase 1 — `dagml-core` (Rust) + minimal Python path.**
- Core: GraphSpec, SearchSpace + lazy enumerator, planner (`find_path`), FoldSet +
  identity splitters, sequential scheduler, PredictionStore (Arrow/Parquet) +
  `oof_join` + aggregation, LineageRecorder, CacheStore, **handle liveness
  manager** (arenas + refcount), **splittable RNG**.
- pyo3 + maturin binding; Python data layer (numpy registry); sklearn controller.
  *DoD: UC6 (stacking) end-to-end in Python, correct OOF, reproducible.*

**Phase 2 — Native blocks.**
- Plug in nirs4all-io (Rust) as loaders; nirs4all-methods (C++) as
  controllers via `extern "C"` shim. Full-native build (without Python).
  *DoD: native pipeline = Python pipeline bit-identical on shared nodes (§7).*

**Phase 3 — Parallelism.**
- Thread scheduler (`GIL_FREE_COMPUTE` controllers); process workers (Python
  GIL-bound, R) with Arrow IPC. *DoD: scaling over folds/variants, determinism
  preserved.*
- Optional multitask batching: the scheduler can group compatible `NodeTask`s
  into a single controller call (`ControllerTaskBatch`) when the manifest
  declares `ACCEPTS_TASK_BATCH` or `ACCEPTS_STATIC_SUBGRAPH`. Target case:
  GPU preprocessing bank known at PLAN, including a Cartesian block compiled
  into a static sub-DAG. *DoD: fixture proving that batched execution produces
  the same outputs/logs/cache/lineage as scalar execution.*

**Phase 4 — R binding.**
- extendr + R data layer + mlr3 controllers (isolated in processes). *DoD: R pipeline = Python pipeline bit-identical on shared C++/control nodes.*

**Phase 5 — Persistence & extras.**
- Bundle (graph + plan + artifacts + schema_fingerprint), PREDICT/EXPLAIN by
  replay, ONNX export, TunerAdapter (Optuna). *DoD: train→export→predict on
  new data, cross-language in inference.*

**Later** — JS/WASM (wasm-bindgen); incremental migration from nirs4all
(cf. spec §19); streaming/out-of-core.

---

## 9. Blockers to watch

| # | Blocker | Mitigation |
|---|---|---|
| 1 | **Handle GC across the DAG** (the hardest) | Scoped arenas + refcount on escapes (working doc Part III); Rust safe for bookkeeping. |
| 2 | **Cross-process marshalling** (GIL-bound / R controllers) | Arrow IPC; bounded to those controllers; local cache at the worker. |
| 3 | **Arrow impedance by type** (graphs/ragged) | Fixed convention, or keep host-local (handle) without crossing. |
| 4 | **FP determinism of C++ methods** | The 5 conditions of §7; internal/Eigen algebra, fixed-order reductions. |
| 5 | **EXPLAIN feature-space** | Opaque-to-core outputs, stored/transmitted uninterpreted. |
| 6 | **Opaque batching too powerful** (a controller hides a topology) | The graph remains explicit; the batch only covers tasks or sub-DAGs known at PLAN, and DAG-ML validates each logical `NodeResult`. |

---

## 10. References

- ABI detail, liveness, describe blob, UC6 trace, RNG, confrontation, Rust/C++ comparison:
  `dag_ml_polyglot_core_design.md`.
- Complete engine contract: `dag_ml_specification_v1.md`.
- 12 materialised use cases: `dag_ml_use_cases.md`.
- Complete data contract: `ml_data_specification_v1.md`.
