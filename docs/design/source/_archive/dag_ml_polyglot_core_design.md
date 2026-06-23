# DAG-ML — Polyglot Core: working document

Status: design in progress (discussion). Companion to `dag_ml_specification_v1.md`
(the ML engine: graph, phases, OOF invariants) and `ml_data_specification_v1.md`
(the data layer). This document explores an **implementation variant**: a native
core (Rust/C++) orchestrating controllers and data that remain in the host language
(Python, R, or native), via a stable ABI.

Purpose of this document: (1) present **the full deliberation** on Python vs
Rust/C++ (both options, their pros/cons/blockers, the middle path, marshalling
boundaries, cross-language portability); (2) freeze the controller ABI; (3) detail
three critical mechanisms — handle liveness, the `describe` blob format, execution
trace of a stacking run; (4) discuss RNG ownership by the core.

Reading note: Part I is not a settled recommendation — it is the **account of the
debate**. It presents the "pure Python + targeted Rust kernels" path *and* the
"polyglot native core" path on equal footing. The choice depends on a single
criterion, stated in §I.13. Parts II→VII detail the polyglot path because it is the
one that requires new design; this is not a verdict against the middle path.

---

## Part I — The Python vs Rust/C++ debate and the direction taken

### I.1 The founding observation

DAG-ML, as specified, **is not a compute kernel — it is a control plane**. It
compiles a DSL into a DAG, enumerates variants, plans and schedules phases
(`COMPILE → PLAN → FIT_CV → SELECT → REFIT → PREDICT → EXPLAIN`), performs OOF
joins, manages lineage/cache/artifacts. Heavy computation (`fit`/`predict`/
`transform`) is **delegated** to operators wrapping sklearn, PyTorch, TF, Keras,
LightGBM, XGBoost (§6.5 of the spec). Those operators *are* host-language objects;
their hot loops are already native (BLAS, CUDA, C++).

Central consequence: **DAG-ML performs almost no FLOPs itself.** The only
computations it owns are the OOF join (§8.2), aggregation (§9.7), variant
enumeration (§4.5), hashing/fingerprint, the control RNG, and the prediction store.
Everything else is I/O and orchestration. **You do not optimize the layer that does
no computing** — but you may want to **port** it beyond Python. It is this dual
observation (little own compute / desire for portability) that structures the entire
debate.

### I.2 Option A — Pure Python

| | |
|---|---|
| **Pros** | • The spec **is already Python**: dataclasses, `Protocol`, `Any`, duck typing (`matches()`). Implementing = transcribing. Minimal time-to-MVP.<br>• Zero-friction integration with the entire operator ecosystem + Optuna/Ray (§10.6). The adapter contract *is* the Python API of those libs.<br>• Migration from nirs4all = incremental refactor (§19), not a rewrite.<br>• numpy/pyarrow/joblib/SQLite provide columnar stores, artifact serialization, and Parquet for free.<br>• Python team, FastAPI webapp → nirs4all: organizational coherence + contributor pool.<br>• Debuggability, introspection, pdb, rich error payloads: natural.<br>• GIL not a real blocker: coarse parallel units (variant/fold), in separate processes (loky §14.3) or native code that releases the GIL. |
| **Cons** | • Pure-Python orchestration overhead (topo loop, dicts in the OOF join, `frozen` dataclasses by the millions) becomes visible **at extreme scale**: 10k+ variants × folds × millions of predictions.<br>• The loky scheduler imposes **pickling** of tasks/results → serialization cost + memory duplication of large `DataBlock`s (mitigated by CoW + Parquet §18).<br>• No compile-time guarantees on invariants: correctness relies on runtime checks (§14.7) + tests.<br>• Memory footprint of Python objects (PredictionBlock/LineageRecord). |
| **Blockers** | • **OOF join (§8.2) and aggregation (§9.7)** written as per-sample Python loops → at high volume, **vectorize** (numpy/pandas/pyarrow groupby). A real blocker but solvable *within* Python.<br>• **Determinism** under loky (§12.3): fix BLAS threads, sort the reducer by `fold_id`. Language-independent.<br>• **Picklability** of `ExecutionPlan`/`NodePlan` (hence `SerializableRef`). Manageable. |

### I.3 Option B — Rust/C++ Core + Python bindings

| | |
|---|---|
| **Pros** | • Control plane structures (GraphSpec, topo sort, `_cartesian_` enumeration, fingerprint, lineage graph): fast, compact, **true threads** (no GIL) for orchestration.<br>• Exhaustive Rust enums → certain invariants encoded **at compile time** (NodeKind, port contracts).<br>• OOF-join / aggregation kernels on **Arrow** (arrow-rs/polars): zero-copy with pyarrow, faster than vectorized numpy.<br>• Memory footprint of millions of lineage/prediction records divided.<br>• Clean ABI for multiple front-ends. |
| **Cons** | • **Operators are in Python.** Any Rust core must call back into Python (pyo3) to execute **every** MODEL/TRANSFORM node → **GIL reacquisition + FFI traversal at every node**. The "no-GIL" benefit evaporates exactly where the work happens.<br>• Hybrid architecture: Rust holds the graph, Python holds operators + arrays. Each `NodeTask` transits ndarrays + operator objects (`Py<PyAny>`). Enormous complexity for **zero compute gain**.<br>• Build/CI explosion: maturin/cibuildwheel, per-platform wheels (linux/mac/win × CPython), manylinux, numpy/pyarrow ABI compat. Heavy tax on a ruff/mypy/pytest CI.<br>• The spec is *Python-shaped* (`operator: Any`, dynamic resolution via `matches()`) → reexpressing it in Rust types **fights the design**.<br>• Reduced team skillset. Time-to-value MVP ≈ **5–10×** for a layer whose runtime is dominated by code we do not own. |
| **Blockers** | • **FFI on opaque operators**: holding/invoking arbitrary Python estimators from Rust (`Py<PyAny>`, GIL token in the executor) → GIL-bound executor at every node.<br>• **Zero-copy exchange**: would require Arrow end-to-end, but operators expect numpy/torch → **reconversion/copy** at every model boundary.<br>• **Artifact serialization**: joblib/torch/tf Python-only → `ArtifactStore` stays Python.<br>• **RNG/SeedContext**: seeds consumed by numpy/torch → Python side.<br>• **Optuna/Ray**: Python-only. |

### I.4 The common blocker, fatal for B in *Python-only deployment*

A Rust/C++ core cannot execute the pipeline without embedding a Python interpreter,
because **all executable value (operators, serialization, tuners, RNG) lives in the
Python orbit**. In Python-only deployment, the result is a Rust orchestrator that
spends its time reacquiring the GIL — the worst of both worlds: Rust complexity
*plus* GIL contention.

Corollary: **the spec has already chosen Python implicitly.** Every type is a
dataclass/Protocol; joblib/SQLite/Parquet/loky/optuna/ray are named throughout.
"Doing Rust" means *rewriting the spec*, not implementing it. This point does not
invalidate B — it invalidates B **if the only target is Python**. The reversal comes
in §I.8.

### I.5 The middle path: Python + targeted Rust kernels

Between A and B, a third path — that of `polars` / `pydantic-core` / `tokenizers`:
surface and orchestration in Python, **hot core in Rust for 3–4 kernels measured as
hot**, never a monolithic core.

1. **Implement DAG-ML in Python** now (orchestration on top of Python ML,
   Python spec, incremental migration §19).
2. **Keep the architecture FFI-friendly.** The only real kernels that DAG-ML owns
   are isolatable behind a `Protocol`:
   - `oof_join` + aggregation (§8.2 / §9.7) — candidate #1, on Arrow
   - enumeration of very large `_cartesian_` spaces (§4.5)
   - canonical fingerprint/hashing (recurring SHA256)
   - optionally the columnar `PredictionStore` (§8.1, already Parquet)
3. **If — and only if — profiling** shows these kernels dominate at scale, replace
   them with a Rust extension (pyo3 + Arrow zero-copy), *behind the same Python
   Protocol*.

This path captures most of the storage gain **without any Rust**: it suffices to
back `PredictionStore`/`FeatureTable` on Arrow/Polars instead of
dataclasses+numpy. Rust only adds value (a) for compute kernels on this Arrow data
and (b) **for non-Python bindings** — which leads to §I.8.

### I.6 The two boundaries never to conflate

The Rust/Python marshalling cost that raises concern is not uniform. We must
separate:

- **Data boundary** (arrays, prediction tables, feature tables): costly marshalling
  **is not inevitable**. Structures with direct sharing (zero-copy) — §I.7.
- **Behavior boundary** (operators: sklearn estimators, torch modules): the call
  cost remains, and no zero-copy structure removes it, because executing host code
  requires the host runtime.

**Data is shareable without copying; behavior is not.** DAG-ML crosses both: arrays
can flow over Arrow/DLPack near-gratis; operator calls remain a GIL call + Python
dispatch per node.

### I.7 Available zero-copy structures, and where copying returns

| Mechanism | Usage | Zero-copy? |
|---|---|---|
| **Apache Arrow C Data Interface** | columnar tables between languages | Yes — two structs (`ArrowSchema`, `ArrowArray`), pointer passing + release callback; stable C ABI, no build dependency. `pyarrow` ↔ `arrow-rs` ↔ R `arrow` ↔ JS. |
| **rust-numpy / buffer protocol (PEP 3118)** | numpy ndarray ↔ `ndarray::ArrayView` | Yes if **C-contiguous**; copy otherwise |
| **DLPack** | torch/tf/jax/cupy tensors, **GPU included** | Yes (`__dlpack__`) |

So for a columnar `PredictionStore`, a `FeatureTable`, OOF arrays: transfer is O(1),
by pointer. Expensive marshalling (pickle, JSON) only concerns boundaries where we
have *not* standardized on Arrow.

**Three leaks where copying returns anyway:**

1. **Columnar (Arrow) vs row-major dense (sklearn).** A matrix
   `(n_samples, n_features)` stored as N Arrow columns → gather/transpose for
   sklearn. Solvable (store as `FixedSizeList`/tensor extension = one buffer),
   but a deliberate choice.
2. **GPU for sklearn** (CPU only; DLPack only covers torch/tf).
3. **Fitted objects never cross in zero-copy** — opaque host state; the core only
   holds a *handle*. At predict time, the core calls back into Python under the GIL.

### I.8 The argument that changes everything: R / JS portability

This is where Option B ceases to be a losing choice. The real driver of B is not
performance — it is **multi-language reach**, and the mechanism is exactly Arrow:

- **Arrow C Data Interface is the polyglot lingua franca**: Rust core ↔ R
  (package `arrow` / `extendr`), ↔ JS (`apache-arrow` / WASM), ↔ Python
  (`pyarrow`). This is how you build a multi-language core (Polars, DataFusion
  model).

But the trap must be named: **a portable control plane does not give a portable
operator ecosystem.**

- **Portable** (and this is the *rigorous core* of the spec): DSL compilation →
  GraphSpec, FoldSet, OOF/leakage invariants, OOF join, lineage, determinism,
  scheduler. A Rust core ports all of this to R and JS.
- **Not portable**: sklearn, torch, lightgbm. In R → tidymodels/mlr3; in JS →
  tfjs/onnx; or re-embed Python. DAG-ML orchestrates operators; the skeleton is
  portable, **the things being orchestrated are not**.

Underestimated benefit: the GIL problem **disappears outside Python** — in R/JS the
operators are native, so the structural call has no GIL to reacquire. The per-node
FFI overhead denounced in §I.4 is *specific to Python*.

### I.9 The polyglot vision

The native core **owns neither data nor operators**. It owns: the graph, the phases,
the folds, the OOF join, lineage, the scheduler, the search space, the control RNG
— **the invariants**. It manipulates **opaque handles** (u64) resolved on the host
side, and sees clearly only the **identity** and **predictions** (in Arrow).

```
        dagml-core (Rust)                      thin bindings
  ┌────────────────────────────┐        ┌──────────────────────────┐
  │ Compiler / GraphSpec        │  pyo3  │ Python: controllers       │
  │ FoldSet, SearchSpace        │◄──────►│   sklearn / torch (handles)│
  │ OOFJoin / PredictionStore   │ extendr│ R: controllers mlr3       │
  │ Lineage / Cache / RNG       │◄──────►│                           │
  │ Scheduler                   │ napi/  │ JS/WASM: tfjs / onnx      │
  │  ── identity+preds = Arrow ─│  wasm  │ native: nirs4all-methods  │
  └────────────────────────────┘        └──────────────────────────┘
   data     : Arrow C Data Interface (zero-copy)
   models   : ONNX / safetensors (cross-language inference)
   control  : core determinism (folds, OOF, RNG)
```

Key difference from Polars: Polars *owns* the data; here the core is **blind** to
it (except identity+predictions). This is what makes it portable without
reimplementing an ML ecosystem per language. `DataBlock`/`FeatureTable`/
`PredictionBlock` cross in zero-copy; only operator handles remain opaque on the
host language side.

### I.10 The GIL: persists, but not a bottleneck; the R asymmetry

- **Structurally**: every call to a Python controller requires the GIL (PyO3
  imposes a `Python<'py>` token). No escape in standard CPython.
- **In throughput**: the core is in Rust and never touches the GIL (graph
  traversal, OOF, lineage, scheduling = free Rust threads). Heavy controllers
  **release** the GIL during BLAS/CUDA. GIL held by `fit` ≈ µs of dispatch;
  compute ≈ ms–s GIL-free. Ratio < 0.1% → near-linear thread scaling. The regime
  where the GIL serializes (nodes ~100 µs) is precisely the one where
  parallelization is not needed. **This is strictly superior** to the spec's
  loky/multiprocess (no pickle, no memory duplication).

**R asymmetry**: R has no GIL but **is not thread-safe** and releases nothing during
computation. R parallelism → **mandatory processes** (fork/PSOCK, cf.
`parallel`/`future`).

| Controllers | Intra-process parallelism | Mechanism |
|---|---|---|
| Python (torch/sklearn) | OK if GIL released | threads + brief GIL |
| R (mlr3) | **no** | processes |
| Native Rust (nirs4all-methods) | **free, total** | threads, no lock |

The fully native build is the **only** one without host-language concurrency
constraints — it is the performance ceiling and the differentiator. Python/R bindings
are "reach/compat" modes with their host's constraints.

### I.11 Three transport channels, three roles

- **Arrow** = transport for *data* (cross-language, zero-copy).
- **ONNX / safetensors** = transport for *fitted models* (cross-language inference).
  Already planned: `ArtifactRef.backend` lists `"onnx"` (§13.1). Do not ask Arrow
  to carry model reproducibility.
- **Native core + control RNG** = transport for *rigor* (folds, OOF,
  lineage — deterministic, and cross-language for the control plane; Part VI).

### I.12 The two principles the ABI encodes

1. **Visibility.** The core sees in Arrow exactly three things: **identity**
   (sample/observation/target/group/origin ids), **predictions** (y_pred/y_proba),
   and **y_true** for scoring. Never X or features. Everything else = handles.
2. **Ownership.** The core owns the **handle lifecycle**; the host owns the
   **underlying object** and frees it on `release`. Arrow carries its own release
   callback (self-describing ownership).

Violating (1) reintroduces marshalling of the heavy stuff. Violating (2) = cross-
language use-after-free or leak driven by the host GC.

### I.13 The decision, restated

This is **not** "Python vs Rust for speed". It is: **do you want a polyglot control
plane?**

- **Essentially Python operators** → **middle path (§I.5)**: pure Python, Arrow as
  internal format for storage gain, surgical Rust kernels if profiling justifies it.
  A Rust core here would cost a GIL hit at every model node for zero compute gain.
- **R/JS reach targeted** → **polyglot native core (§I.9)**: data marshalling is
  not the obstacle (Arrow zero-copy); the GIL disappears outside Python; you get
  the **same rigorous ML engine in Python, R, and JS**. Cost: multi-platform CI,
  plugin ABI to freeze, and rewriting the spec rather than implementing it.

The rest of this document (Parts II→VII) details the polyglot path.

---

## Part II — The ABI

### 8. Substrate

- **C vtable** (`#[repr(C)]`, function pointers) — the GCD that PyO3, extendr, and
  native Rust fill identically.
- **Arrow C Data Interface** for everything the core reads.
- **Versioned canonical blobs** (`Bytes`) for the rich-but-evolving (params,
  `ModelInputSpec`, descriptors, errors). Never a C struct for this → the ABI
  does not change when the spec evolves.

### 9. `ControllerVTable`

```rust
// ───────────────────────── boundary types ─────────────────────────
#[repr(C)] pub struct Bytes { ptr: *const u8, len: usize }      // ownership annotation below
#[repr(u8)] pub enum Phase { Compile, Plan, FitCv, Select, Refit, Predict, Explain }
#[repr(u8)] pub enum Status { Ok, Skip, Error }
#[repr(u8)] pub enum Backend { Joblib, Torch, Onnx, Safetensors, Json, Raw } // = ArtifactRef.backend §13.1
pub type HandleId = u64;                                         // 0 = null

#[repr(C)] pub struct ArrowSchema { /* Arrow CDI */ }
#[repr(C)] pub struct ArrowArray  { /* Arrow CDI ; carries its own release */ }
#[repr(C)] pub struct ArrowOut { schema: *mut ArrowSchema, array: *mut ArrowArray } // [owned→core]
#[repr(C)] pub struct NamedHandle { port: Bytes, handle: HandleId }                  // multi-source input

// capabilities (bitset) — merges AdapterSpec.capabilities §6.1 + ResourceHints §2.4
pub const SUPPORTS_PREDICT:      u64 = 1<<0;
pub const SUPPORTS_PROBA:        u64 = 1<<1;
pub const SUPPORTS_EXPLAIN:      u64 = 1<<2;
pub const STATEFUL:              u64 = 1<<3;   // fit -> fitted handle ; otherwise stateless
pub const INVERTIBLE:            u64 = 1<<4;   // y_transform
pub const GIL_FREE_COMPUTE:      u64 = 1<<5;   // releases GIL during compute -> thread-schedulable
pub const THREAD_SAFE:           u64 = 1<<6;   // concurrent instances OK
pub const DETERMINISTIC:         u64 = 1<<7;   // (seed/stream) reproducible -> cache valid
pub const REQUIRES_DATASET_PLAN: u64 = 1<<8;   // deferred planning to FIT_CV §5.3
pub const EMITS_RELATION:        u64 = 1<<9;   // changes the sample set (augmentation)
pub const RNG_FROM_CORE:         u64 = 1<<10;  // draws all randomness from the core RNG (Tier 1, Part VI)

// ───────────────────────── call context (core → host) ─────────────────────────
#[repr(C)] pub struct CallContext {
    phase: Phase,
    run_id: Bytes, variant_id: Bytes, node_id: Bytes,           // [borrowed]
    fold_id: Bytes,                                             // "0".."K-1"|"final"|""
    branch_path: Bytes,                                         // canonical tuple
    rng_stream: u64,                                            // RNG stream derived by the core (Part VI)
    rng_seed_legacy: u64,                                       // derived integer seed (Tier 2, frameworks)
    callbacks: *const CoreCallbacks,
}
#[repr(C)] pub struct CoreCallbacks {                           // upcalls host → core (minimized)
    ctx: *mut c_void,
    log_metric:      extern "C" fn(*mut c_void, name: Bytes, value: f64),
    report_progress: extern "C" fn(*mut c_void, done: u64, total: u64),
    check_cancel:    extern "C" fn(*mut c_void) -> bool,
    rng_fill_f64:    extern "C" fn(*mut c_void, stream: u64, n: u64) -> ArrowOut,   // core draw (Tier 1)
    rng_permutation: extern "C" fn(*mut c_void, stream: u64, n: u64) -> ArrowOut,   // core permutation
}

// ───────────────────────── result envelopes ─────────────────────────
#[repr(C)] pub struct FitResult {
    status: Status,
    fitted: HandleId,        // [handle, owned→host] 0 on error/skip
    metrics: ArrowOut,       // [owned→core] nullable
    error: Bytes,            // [owned→host] ErrorPayload §14.6 if Error
}
#[repr(C)] pub struct PredictResult {
    status: Status,
    predictions: ArrowOut,   // [owned→core] y_pred (+ proba as columns), self-describing schema
    target_space: Bytes,     // [owned→host] "raw"|"scaled"|...
    metrics: ArrowOut,       // [owned→core] nullable
    error: Bytes,
}
#[repr(C)] pub struct TransformResult {
    status: Status,
    fitted: HandleId,        // [handle] 0 if stateless
    output: HandleId,        // [handle, owned→host] transformed X — NEVER read by the core
    relation: ArrowOut,      // [owned→core] nullable ; non-null if EMITS_RELATION (origin_id…)
    error: Bytes,
}

// ───────────────────────── the vtable ─────────────────────────
#[repr(C)] pub struct ControllerVTable {
    abi_version: u32,
    state: *mut c_void,            // controller instance / host closure env
    kind: u8,                      // NodeKind served (§2.1)
    capabilities: u64,

    // — lazy construction: the core drives WHICH params (search space), the host HOW —
    clone_with: extern "C" fn(state: *mut c_void, operator: HandleId, params: Bytes) -> HandleId,

    // — PLAN introspection (without data) —
    describe: extern "C" fn(state: *mut c_void, operator: HandleId) -> Bytes,   // [owned→host] cf. §13
    matches:  Option<extern "C" fn(state: *mut c_void, kind: u8, operator: HandleId) -> bool>,

    // — SPLIT (identity; optional data handle for feature-based splitters, e.g. KS/SPXY) —
    split: Option<extern "C" fn(state: *mut c_void, operator: HandleId,
                                identity: *mut ArrowArray, identity_schema: *mut ArrowSchema, // [borrowed]
                                y: *mut ArrowArray, y_schema: *mut ArrowSchema,                // nullable
                                data: HandleId,                                                // 0 except feature-based
                                ctx: *const CallContext) -> ArrowOut>,  // (sample_id, fold_id, partition)

    // — FIT-like — (target = handle: resolved host-side, never marshalled into the controller) —
    fit: Option<extern "C" fn(state: *mut c_void, operator: HandleId,
                              inputs: *const NamedHandle, n_inputs: usize, target: HandleId,
                              ctx: *const CallContext) -> FitResult>,
    transform_fit: Option<extern "C" fn(state: *mut c_void, operator: HandleId,
                                        input: HandleId, ctx: *const CallContext) -> TransformResult>,

    // — APPLY-like —
    predict: Option<extern "C" fn(state: *mut c_void, fitted: HandleId,
                                  inputs: *const NamedHandle, n_inputs: usize, want_proba: bool,
                                  ctx: *const CallContext) -> PredictResult>,
    transform_apply: Option<extern "C" fn(state: *mut c_void, fitted: HandleId,
                                          input: HandleId, ctx: *const CallContext) -> TransformResult>,
    invert: Option<extern "C" fn(state: *mut c_void, fitted: HandleId,
                                 y_in: *mut ArrowArray, y_schema: *mut ArrowSchema) -> ArrowOut>,

    // — EXPLAIN (leaf; output opaque to the core) —
    explain: Option<extern "C" fn(state: *mut c_void, fitted: HandleId,
                                  inputs: *const NamedHandle, n_inputs: usize,
                                  cfg: Bytes, ctx: *const CallContext) -> ArrowOut>,

    // — persistence / cross-language inference —
    serialize:   extern "C" fn(state: *mut c_void, fitted: HandleId, backend: Backend) -> Bytes,
    deserialize: extern "C" fn(state: *mut c_void, blob: Bytes, backend: Backend) -> HandleId,

    // — cache (optional override; otherwise the core computes the key §13.3) —
    cache_key: Option<extern "C" fn(state: *mut c_void, operator: HandleId,
                                    inputs: *const NamedHandle, n_inputs: usize,
                                    ctx: *const CallContext) -> Bytes>,

    // — lifecycle —
    release:    extern "C" fn(state: *mut c_void, handle: HandleId),
    free_bytes: extern "C" fn(state: *mut c_void, b: Bytes),
    destroy:    extern "C" fn(state: *mut c_void),
}
```

Mapping to spec §6: `describe` ⊃ `input_spec`+`aux_inputs`+`declare_ports`+
`supported_phases`; `fit`/`predict`/`predict_proba` → `fit`/`predict(want_proba)`;
`transform_*` → `TransformerMixin`; `invert` → `inverse_transform`;
`cache_key` → same. `clone_with` and `split` are necessary ABI refinements
revealed by the trace (Part V).

### 10. `DataVTable` (mandatory counterpart)

The controller signature is meaningless without defining who creates/resolves
handles and who exposes identity to the core.

```rust
#[repr(C)] pub struct DataVTable {
    abi_version: u32, state: *mut c_void,
    materialize:   extern "C" fn(state, source_id: Bytes, ctx: *const CallContext) -> HandleId,
    view_identity: extern "C" fn(state, h: HandleId) -> ArrowOut,  // sample/obs/target/group/origin ids
    make_view:     extern "C" fn(state, h: HandleId,               // slice by sample-ids (folds §7.3)
                                 ids: *mut ArrowArray, sch: *mut ArrowSchema, partition: u8) -> HandleId,
    target_arrow:  extern "C" fn(state, h: HandleId) -> ArrowOut,  // y_true for scoring (core)
    feature_arrow: extern "C" fn(state, h: HandleId) -> ArrowOut,  // X/features per observation
    ingest_arrow:  extern "C" fn(state, *mut ArrowArray, *mut ArrowSchema) -> HandleId, // OOF -> handle
    handle_nbytes: extern "C" fn(state, h: HandleId) -> u64,       // step-cache budget §13.4
    schema_fingerprint: extern "C" fn(state, h: HandleId) -> Bytes,
    release: extern "C" fn(state, h: HandleId), destroy: extern "C" fn(state),
}
```

Key point: `make_view` slices **by sample-ids**, never by positions —
the anti-leakage invariant (§7.3) is enforced *by the ABI itself*. The controller
receives an already-sliced handle; it cannot look outside the fold.

### 11. All three implementations fill the **same** struct

- **Python (PyO3)**: each fn acquires the GIL, resolves `handle→object` in a host
  registry, calls `estimator.fit(X, y)`, re-registers the fitted object, returns a
  handle. `clone_with` = `sklearn.clone` + `set_params`. `serialize(Onnx)` =
  `skl2onnx`/`torch.onnx`.
- **R (extendr)**: same; R evaluator is not thread-safe → `THREAD_SAFE=0`,
  `GIL_FREE_COMPUTE=0` for all → scheduler routes via **processes**.
- **Native Rust (nirs4all-methods)**: the fns *are* Rust; handle = index into a
  slab; `GIL_FREE_COMPUTE=1`, `THREAD_SAFE=1`; zero marshalling, zero indirection
  beyond the vtable pointer.

---

## Part III — (a) Handle liveness protocol

Blocker #1. The ABI makes ownership explicit (`release` + Arrow release), but
**correctness of liveness tracking** across branches/folds/sub-DAGs is the core's
responsibility. Bug = use-after-free (release too early) or leak (never released,
driven by the host GC).

### a.1 The two naive approaches

| Approach | Principle | + | − |
|---|---|---|---|
| **Per-edge refcount** | refcount = number of consumers; decrement at each completed consumer; release at 0 | prompt release, minimal memory | branches/folds/map multiply consumers; cache complicates; skips to handle |
| **Phase/scope sweep** | every handle from a generation lives until the end of the scope, then bulk release | simple, robust | memory = full scope peak |

### a.2 The chosen synthesis: scoped arenas + refcount on escaped handles

Each fold/branch/variant execution is an **arena**. Rule:

- A handle created in an arena and **not escaped** is bulk-released at arena close
  (sweep, simple and infallible).
- A handle that **escapes** the arena (e.g. a base model surviving to REFIT; an OOF
  prediction feeding the join; a cached handle) is **refcounted** and promoted to
  the parent scope.

**Refcount invariant of an escaped handle:**

```
refcount(h) = (# consumer nodes not yet executed in the plan, h as input)
            + (1 if h is referenced by the CacheStore)
            + (1 if h is promoted to the bundle / REFIT)
```

When `refcount(h)` reaches 0 → `vtable.release(state, h)`.

### a.3 Trigger points

| Event | Action |
|---|---|
| Node completed | for each input handle: if escaped, decrement; otherwise leave alone (the arena handles it) |
| Node *skipped* (phase not supported) or in error | decrement its input refs anyway (otherwise leak) |
| Arena close (end of fold/branch) | sweep: release all local non-escaped handles |
| Handle put in cache | +1 ref (cache) → survives the arena |
| LRU eviction (budget `step_cache_max_mb` via `handle_nbytes`) | −1 ref (cache); release if 0 |
| Promotion to bundle (REFIT) | +1 ref; freed after bundle export |

### a.4 Knowing the consumer count

From `ExecutionPlan.topological_order` + outgoing edges per port. For dynamic
fan-out (`MapNode` over branches), the count is known **after fork expansion**.
For folds, each fold is a distinct consumer instance.

### a.5 Cross-process

Refcounts are **per-process**. A worker (variant) opens an arena, executes it, and
on return: `destroy` its vtables → bulk-free everything. Results returned to the
orchestrator are **Arrow** (already owned→core), so cross-process liveness is trivial
(bounded to the worker's lifetime). The step-cache is then **local to the worker**
— only useful for batches of variants assigned to the same worker (acceptable
degradation).

### a.6 Micro-example (one fold of UC6, branch b0)

```
arena fold0 opened
  Hd:train0  = make_view(Hd:nir, ids_train0, train)      # arena local
  Hd:val0    = make_view(Hd:nir, ids_val0,   val)         # arena local
  Hd:Xt_tr   = transform_fit(SNV, Hd:train0).output       # local (SNV stateless)
  Hd:Xt_val  = transform_apply(SNV, Hd:val0).output        # local
  Hf:pls0    = fit(PLS, [X:Hd:Xt_tr], y_tr).fitted         # ESCAPES -> refcount=2 (REFIT? no; predict0 + ?)
  A:pred_val = predict(Hf:pls0, [X:Hd:Xt_val])             # Arrow -> PredictionStore (core owned)
  # Hf:pls0 : consumed by predict0 -> decrement ; no other consumer in CV -> release
  #   (at REFIT, a NEW Hf:pls_final will be fit on full train; the fold's pls0 does not survive)
arena fold0 close -> sweep: release Hd:train0, Hd:val0, Hd:Xt_tr, Hd:Xt_val
```

Only `A:pred_val` (Arrow, small) survives, in the `PredictionStore`. All X handles
from the fold are swept. Memory peak = one fold at a time.

---

## Part IV — (b) `describe` blob format

`describe(operator) -> Bytes` is the **PLAN contract**. Encoding: canonical JSON
(sorted keys, UTF-8) prefixed with a version tag `"dagml.describe/1"`. JSON chosen
for v1: debuggable, stable, available in every language. (Msgpack possible if
compactness required; the logical schema is identical.)

### b.1 Logical schema

```jsonc
{
  "v": "dagml.describe/1",
  "adapter": { "id": "sklearn.estimator", "version": "1.0.0", "kind": "model" },
  "phases": ["fit_cv", "refit", "predict"],
  "capabilities": ["supports_predict", "stateful", "deterministic",
                   "gil_free_compute", "thread_safe"],
  "ports": {
    "inputs":  [{ "name": "X", "kind": "data",
                  "representation": "tabular_numeric", "cardinality": "one" }],
    "outputs": [{ "name": "y_pred", "kind": "prediction",
                  "representation": "tabular_numeric", "cardinality": "one" }]
  },
  "input_spec": {                       // = ModelInputSpec (ml_data.contract)
    "representation": "tabular_numeric",
    "rank": 2, "dtype": "float32", "layout": "row_major",
    "required_sources": ["*"],          // "*" = any merged source; otherwise names
    "accepts_missing": false,
    "max_features": null
  },
  "aux_inputs": [],                     // e.g. [{"name":"wavelengths","representation":"axis_coords","required":false}]
  "planning": {
    "requires_dataset_at_plan": false,  // true -> data_plan deferred to FIT_CV (§5.3)
    "allow_lossy": false,
    "fit_scope": "fold_train"           // fold_train | train_once | stateless
  },
  "identity_params": ["n_components"],  // params that enter the fingerprint/cache
  "resources": {                         // = ResourceHints (§2.4)
    "cpu": 1, "gpu": 0, "memory_mb": null,
    "thread_safe": true, "nested_parallelism": "forbid", "timeout_seconds": null
  }
}
```

### b.2 Example — PyTorch CNN (contrast)

```jsonc
{
  "v": "dagml.describe/1",
  "adapter": { "id": "pytorch.module", "version": "1.0.0", "kind": "model" },
  "phases": ["fit_cv", "refit", "predict", "explain"],
  "capabilities": ["supports_predict", "supports_explain", "stateful"],   // NOT gil_free during pure Python fit
  "ports": { "inputs":  [{ "name": "X", "kind": "data",
                           "representation": "signal_with_processings", "cardinality": "one" }],
             "outputs": [{ "name": "y_pred", "kind": "prediction",
                           "representation": "tabular_numeric", "cardinality": "one" }] },
  "input_spec": { "representation": "signal_with_processings",
                  "rank": 3, "dtype": "float32", "layout": "row_major",
                  "required_sources": ["*"], "accepts_missing": false, "max_features": null },
  "aux_inputs": [],
  "planning": { "requires_dataset_at_plan": true,    // architecture depends on input dim -> deferred
                "allow_lossy": false, "fit_scope": "fold_train" },
  "identity_params": ["arch", "lr", "epochs", "batch_size"],
  "resources": { "cpu": 4, "gpu": 1, "gpu_memory_mb": 4096,
                 "thread_safe": false, "nested_parallelism": "forbid", "timeout_seconds": 3600 }
}
```

### b.3 Fields that drive a core decision

| Field | Core decision |
|---|---|
| `requires_dataset_at_plan` | if `true` → `data_plan=None` at PLAN, re-resolved in fold scope before fit (§5.3) |
| `fit_scope` | `fold_train` = refit per fold (correct OOF, expensive); `train_once` = single fit on frozen full train for CV (slight leakage, UC1 friction #4 case); `stateless` = no fitted object |
| `capabilities: gil_free_compute / thread_safe` | choice of threads vs processes (§I.10) |
| `identity_params` | fingerprint input + cache key (§13.3) |
| `resources` | scheduler: overcommit, nested parallelism, timeout |
| `allow_lossy` | refusal / `requires_user_choice` escalation at PLAN (UC1) |

---

## Part V — (c) Execution trace: UC6 (3-way stacking + Ridge meta-learner)

Pipeline (recap): `nir(500) → y standardize → KFold(5,rs42) outer → branch[
SNV+PLS(12) | MSC+RF(300) | Detrend+SVR(rbf,C10) ] → merge predictions(validate
OOF) → KFold(5,rs42) inner → Ridge(1.0)`.

Notation: `[Ho:x]` operator handle · `[Hd:x]` data handle · `[Hf:x]` fitted handle ·
`[A:x→core]` Arrow owned by core · `vt.X` = vtable call.

### COMPILE (host front-end → core)

```
HOST: parse DSL, register operators in the host registry, emit a neutral ProtoGraph:
      SNV→[Ho:1] PLS12→[Ho:2] MSC→[Ho:3] RF300→[Ho:4] Detrend→[Ho:5] SVR→[Ho:6]
      Ridge→[Ho:7] yStd→[Ho:8] KFouter→[Ho:9] KFinner→[Ho:10]
CORE: GraphSpec, check acyclicity + port arities. No search space here -> 1 variant.
```

### PLAN

```
CORE: for each data-aware operator:
      vt.describe([Ho:2]) -> {model, tabular_numeric, rank2, fit_scope=fold_train, identity_params=[n_components]}
      vt.describe([Ho:4]) -> {model, ..., capabilities without rng_from_core (RF bootstrap = Tier 2)}
      vt.describe([Ho:6]) -> {model, ..., deterministic (SVR without RNG)}
      KFold (identity only) -> NATIVE core splitter, no vt.split call (Tier 1, see Part VI)
      resolve DataPlans (tabular_numeric), schema_fingerprint.
```

### FIT_CV — level 0 (base learners)

```
CORE: data.materialize("nir") -> [Hd:nir]   ; data.view_identity([Hd:nir]) -> [A:ids 500]
CORE: NATIVE outer split with core RNG(stream derived from path "split:outer") -> [A:folds_outer]   # Tier 1
```

For **fold0** (folds 1–4 identical, each with its own sub-stream from its path):

```
arena fold0:
  data.make_view([Hd:nir], ids_train0, train) -> [Hd:tr0]
  data.make_view([Hd:nir], ids_val0,   val)   -> [Hd:val0]

  # y standardize, fit on train (per-fold, anti-leakage), invert kept for scoring
  vt.transform_fit([Ho:8], target_handle(tr0)) -> [Hf:ystd0], y_scaled_train (host-side)

  # ── branch b0: SNV + PLS ──
  vt.transform_fit([Ho:1], [Hd:tr0])  -> ([Hf:0 stateless], [Hd:Xt_tr_b0])      # SNV stateless
  vt.transform_apply([Ho:1], [Hd:val0]) -> [Hd:Xt_val_b0]
  vt.fit([Ho:2], inputs=[(X,[Hd:Xt_tr_b0])], target=y_scaled_train, ctx{fold0}) -> [Hf:pls0]
  vt.predict([Hf:pls0], inputs=[(X,[Hd:Xt_val_b0])], want_proba=false) -> [A:pred_b0_scaled→core]
  vt.invert([Hf:ystd0], [A:pred_b0_scaled]) -> [A:pred_b0_raw→core]
  CORE: PredictionStore.append(producer=b0, fold=0, partition=val, ids=ids_val0, y_pred=pred_b0_raw)

  # ── branch b1: MSC (stateful) + RF (bootstrap = Tier 2) ──
  vt.transform_fit([Ho:3], [Hd:tr0]) -> ([Hf:msc0], [Hd:Xt_tr_b1])              # MSC learns mean spectrum
  vt.transform_apply([Hf:msc0], [Hd:val0]) -> [Hd:Xt_val_b1]
  vt.fit([Ho:4], [(X,[Hd:Xt_tr_b1])], y_scaled_train, ctx{rng_seed_legacy})     # RF: random_state = Tier 2 seed
       -> [Hf:rf0]
  vt.predict([Hf:rf0], [(X,[Hd:Xt_val_b1])]) -> [A:pred_b1_scaled→core]
  vt.invert([Hf:ystd0], [A:pred_b1_scaled]) -> [A:pred_b1_raw→core] ; append(b1,fold0,val)

  # ── branch b2: Detrend + SVR (deterministic) ──
  vt.transform_fit([Ho:5], [Hd:tr0]) -> ([Hf:0], [Hd:Xt_tr_b2])
  vt.transform_apply([Ho:5], [Hd:val0]) -> [Hd:Xt_val_b2]
  vt.fit([Ho:6], [(X,[Hd:Xt_tr_b2])], y_scaled_train) -> [Hf:svr0]
  vt.predict([Hf:svr0], [(X,[Hd:Xt_val_b2])]) -> [A:pred_b2_scaled→core]
  vt.invert([Hf:ystd0], [A:pred_b2_scaled]) -> [A:pred_b2_raw→core] ; append(b2,fold0,val)

arena fold0 close -> sweep: release [Hd:tr0],[Hd:val0],[Hd:Xt_*],[Hf:pls0],[Hf:rf0],[Hf:svr0],
                                    [Hf:msc0],[Hf:ystd0]   # nothing escapes in CV level 0
```

After 5 folds: `PredictionStore` contains 3 producers × 500 OOF (each sample
predicted once, in its validation fold).

### join:pred — `oof_join` (100% core, Rust, on Arrow)

```
CORE (no vtable calls):
  for b0,b1,b2: verify coverage of all 500 ids in partition=val
                verify ABSENCE of partition=train  -> otherwise OOFLeakageError (allow_train_predictions=false)
  build [A:meta_features→core]: columns (b0_pls,b1_rf,b2_svr), 500 rows, indexed by sample_id
  data.ingest_arrow([A:meta_features]) -> [Hd:meta]      # OOF re-enters as native data
```

### FIT_CV — level 1 (meta-learner)

```
CORE: NATIVE inner split, core RNG(path "split:inner") -> [A:folds_inner]     # Tier 1, same 500 samples
for each inner fold j:
  arena inner_j:
    data.make_view([Hd:meta], ids_tr_j, train) -> [Hd:meta_tr_j]
    data.make_view([Hd:meta], ids_val_j, val)  -> [Hd:meta_val_j]
    vt.fit([Ho:7], [(X,[Hd:meta_tr_j])], target=y_train_j) -> [Hf:ridge_j]
    vt.predict([Hf:ridge_j], [(X,[Hd:meta_val_j])]) -> [A:meta_pred_j→core]
    PredictionStore.append(producer=ridge, fold=j, val)
  close -> sweep
```

### SELECT

```
CORE: rmsecv_meta from meta OOF (500) + y_true (data.target_arrow). 1 variant -> selected. SelectionRecord.
```

### REFIT (fold_id="final")

```
CORE: open arena "final"
  # base learners refit on FULL train (500)
  for b in {b0,b1,b2}:
     vt.transform_fit(preproc_b, [Hd:nir_fulltrain]) -> Xt_full_b
     vt.fit(model_b_op, [(X,Xt_full_b)], y_full) -> [Hf:base_b_final]   # ESCAPES -> promoted to bundle (ref+1)
  # meta refit on OOF (NOT on refit base preds -> otherwise train leak)
  vt.fit([Ho:7], [(X,[Hd:meta])], y_full) -> [Hf:ridge_final]            # ESCAPES -> promoted to bundle
  # portable serialization
  for h in {base0,base1,base2,ridge}_final: vt.serialize(h, Onnx|Joblib) -> bytes -> ArtifactStore
  export bundle: graph + plan + 4 artifacts + 3 OOF caches + schema_fingerprint
  release promoted handles after export
```

### Reading the trace

- **Host-side handles**: all transformed X, all fitted objects. Never cross over.
- **Arrow → core**: folds, predictions, y_true, OOF table, identity. All small.
- **The only heavy materialization**: `ingest_arrow(meta_features)` — but that is
  500×3, negligible.
- **Leakage validation**: 100% core, without touching the host (Rust on Arrow).
- **Two RNG tiers visible**: KFold via core RNG (Tier 1); RF bootstrap via
  `rng_seed_legacy` (Tier 2). See Part VI.
- **Meta refit on OOF, not on refit predictions**: anti-leakage invariant enforced
  by the core, not the controller.

---

## Part VI — Discussion: RNG in the library (the core)

Question: can we have the **core own the RNG** rather than delegating to the host
language's RNG (numpy/R/torch)? The stakes: cross-language reproducibility, which
has failed until now because `numpy.random ≠ R RNG ≠ torch`.

### VI.1 What is random in an ML pipeline?

| Source of randomness | Space | Ownable by the core? |
|---|---|---|
| Fold splits / shuffles | indices / identity | **Yes** — already on the core side |
| Tuner sampling (random/Bayes) | params | **Yes** — already core side (search space) |
| Bootstrap (RF): index draws | indices | **Yes in principle** (indices), **no in practice** (sklearn does not allow injecting an external stream) |
| Augmentation: *selection* (which samples, mixup pairs, coefficients) | indices / scalars | **Yes** — small, pre-drawable |
| Augmentation: *noise tensor* (feature-shape) | feature-space | **No** without seeing the dims (large) |
| NN weight init / dropout masks | framework parameters | **No** — internal to torch/tf, non-redirectable |

**Key**: the core can own the RNG for the entire **control plane**
(indices/identity/params) — precisely where ML rigor demands it. It cannot own the
RNG **internal to frameworks** (weight init).

### VI.2 The enabling technology: counter-based, splittable PRNG

A **counter-based** PRNG (Philox, Threefry) or **splittable** PRNG (SplitMix, the
JAX-style PRNG) is:

- **Portable and reproducible across languages by construction** — pure integer
  arithmetic, defined rounds/constants: same key+counter → same bits everywhere
  (unlike Mersenne Twisters whose *seeding* differs). This is exactly why JAX uses
  Threefry and why numpy added Philox/PCG.
- **Splittable**: each path (run, variant, node, fold, branch, aug_index) derives
  an independent, deterministic sub-key **without coordination**.
- **O(1) access** (counter-based) → trivially parallel.

### VI.3 Rewriting `SeedContext` as a splittable PRNG tree

Currently `SeedContext.derived() = SHA256(root || path) mod 2³²` (§12) — loses
entropy, is not a stream. Proposal:

```
key(path)  = SHA256(canonical_utf8(path_labels))[:16]      # 128-bit, portable
stream     = Philox(key=key(path), counter=0)              # independent stream per path
```

`SeedContext.child(...)` becomes a PRNG *split*. The `CallContext` carries a
`rng_stream: u64` (stream id resolved by the core) and, for non-redirectable
frameworks, a `rng_seed_legacy: u64` (integer derived from the same stream).

### VI.4 Two tiers of reproducibility (to be stated explicitly)

- **Tier 1 — control RNG, owned by the core, cross-language bit-identical.**
  Splits, tuner sampling, augmentation *selection*, bootstrap *when* the operator
  accepts an external stream. The controller declares `RNG_FROM_CORE` and draws its
  randomness via the `rng_fill_f64` / `rng_permutation` upcalls, **or** the core
  **pre-draws** the values and passes them in Arrow (the split case: the core does
  the split natively and passes `[A:folds]`).

- **Tier 2 — RNG internal to the framework, owned by the host, reproducible
  intra-lib only.** Weight init, dropout, sklearn bootstrap. The core passes
  `rng_seed_legacy`; reproducible only with equal lib/version/platform.

The combination `RNG_FROM_CORE + DETERMINISTIC` ⇒ cross-language bit identity. Its
absence ⇒ Tier 2.

### VI.5 Passing mechanism (efficiency)

- **Small draws** (permutations, selections, coefficients): the core **pre-draws
  into Arrow**. Always bit-identical; the host needs no RNG. Covers splits, tuner,
  augmentation selection, bootstrap indices.
- **Feature-shaped draws** (large noise tensor): either `rng_fill` upcall in chunks
  (Tier 1, call overhead), or accepted as Tier 2. v1: **control = Arrow pre-draw
  (Tier 1)**; **internal framework = seed (Tier 2)**.

### VI.6 The unexpected gain: determinism independent of scheduling

Since a splittable PRNG determines the stream **by the path, not by execution
order**, the result is identical whether folds run sequentially, in threads, in loky,
or in Ray. The point §12.3.3 of the spec ("the scheduler does not change the order
of results") becomes **free** for all control randomness. This is a strong argument
on top of portability.

### VI.7 Splitter cases

- **Identity-only splitters** (KFold, ShuffleSplit, GroupKFold, Stratified: read
  only ids/y/groups) → **native to the core**, core RNG, Tier 1, cross-language.
  No `vt.split` call.
- **Feature-based splitters** (KennardStone, SPXY: spectral distances → need X) →
  host controllers via `vt.split(..., data=handle)`. They are **deterministic**
  (greedy, no RNG); their reproducibility is a problem of numerical determinism
  (distance ties), not RNG.

### VI.8 Honest limits

- Framework internals (torch init, sklearn bootstrap) remain Tier 2:
  **a fundamental boundary, not a design failure**.
- Each Tier 1 controller must accept its randomness as input (Arrow) rather than
  calling its native RNG — **adaptation required**, especially for augmentation.
  Identity splitters are free (the core handles them).
- Two tiers to document, and a flag (`RNG_FROM_CORE`) to keep honest, or risk a
  false reproducibility promise.

### VI.9 Proposed decision

Put a **counter-based splittable PRNG in the core** (Philox/Threefry) and rewrite
`SeedContext` as a stream tree. Cover the control plane in Tier 1 (Arrow pre-draw).
Retain `rng_seed_legacy` for Tier 2. Document both tiers as the reproducibility
contract.

### VI.10 Consequence: cross-language reproducibility with nirs4all-methods

Concrete question: with the validated portable C++ methods (**nirs4all-methods**),
can we have a Python pipeline — to which a pure Python model (torch, sklearn) can
optionally be attached — and an R pipeline, that give **exactly the same results**,
except for the language-specific models?

**Yes** — and the decisive asset is precisely **owning nirs4all-methods in C++**.
Breakdown by node type:

| Node type | Python | R | Identical between the two? |
|---|---|---|---|
| Preprocessing / model **nirs4all-methods (C++)** | same C++ binary | same C++ binary | **Yes, bit-identical** |
| Control plane (folds, OOF, selection, randomness) | core + RNG Tier 1 | core + RNG Tier 1 | **Yes, bit-identical** |
| **Pure-language** model (torch/sklearn / mlr3) | present | absent or different | **No — this is the exception** |

Why this is solid: it is *your* code. The same C++ binary, called from the Python
or R binding, on the **same data** (nirs4all-io C++) and the **same control
randomness** (core RNG, Tier 1), produces **bit-identical** results — a guarantee
that numpy/torch/native-R never provide, since their compilation is not under our
control.

**Localizing the divergence.** It is confined to language-specific model nodes. A
torch model in Python has no R twin → at that node, the two pipelines compute
different things. This is not non-reproducibility: these are **literally two
different pipelines at that node** (Tier 2). Concretely:

- Python `[C++ methods only]` ≡ R `[same C++ methods only]` → **identical end to
  end**.
- Python `[C++ methods + torch]` vs R `[same C++ methods + mlr3]` →
  **bit-identical up to the model node**, then divergence (by design).
- To make the model node *also* identical cross-language: (a) a native
  **nirs4all-methods** model (C++, shared), or (b) export the fitted model as
  **ONNX** and replay **inference** (deterministic) in the other language — but
  those are the *same trained weights*, not an independently retrained model.

**The 5 conditions for the word "exactly".** Bit-identity of C++ nodes rests on
floating-point determinism that you, specifically, are the only one who can
guarantee:

1. **Coherent compilation of both bindings** — ideally the *same* shared lib,
   otherwise same toolchain/flags. No divergent `-ffast-math` between the Python
   wheel and the R package.
2. **Deterministically ordered reductions** — the classic trap: a parallel sum
   reorders floats → differences in the last ULPs, amplified in iterative
   algorithms (NIPALS/SVD in PLS). Single-thread or fixed-order reductions.
3. **Controlled linear algebra** — if the methods call a *system* BLAS/LAPACK,
   Python and R may link two different ones (OpenBLAS vs MKL, different threads)
   → divergence. Internal algebra or Eigen (header-only, deterministic at equal
   compilation) eliminates the risk.
4. **Control randomness exclusively from the core** (`RNG_FROM_CORE` + Arrow
   pre-draw) — no controller draws with its own RNG for a Tier 1 decision.
5. **Same ingestion** (nirs4all-io C++) and **same dtype** for predictions in the
   Arrow schema.

These conditions are **under your control**, unlike the Python ecosystem. Met, they
give: a Python pipeline and an R pipeline **bit-identical on every shared node
(C++ + control)**, with the only divergence being — exactly — the models specific to
each language. This is the contract that the Tier 1 / Tier 2 separation and the
native core make achievable.

---

## Part VII — Confrontation with objectives & remaining blockers

| Objective | Verdict | Note |
|---|---|---|
| Core blind to data except identity+predictions | ✅ | handles for X/features/fitted; Arrow for identity, predictions, y_true, relation |
| Same Python/R/native signature | ✅ | C vtable; only the implementations differ |
| GIL non-blocking in throughput | ✅ conditional | `GIL_FREE_COMPUTE` knob routes thread vs process; native build short-circuits |
| "fit handle → handle + preds Arrow" | ✅ | `fit→FitResult{fitted}` then `predict→PredictResult{predictions}` |
| Reproducibility | ⚠️ scoped → ✅ Tier 1 | core RNG makes the **control plane** cross-language bit-identical; framework internals = Tier 2; inference = ONNX |
| Native nirs4all batteries | ✅ | vtable in Rust: zero GIL, zero marshalling, free parallelism = performance ceiling |
| ABI stability vs spec evolution | ✅ | rich/evolving content in versioned `Bytes`; vtable extensible via bits + optional fns |

### Remaining blockers (in risk order)

1. **Handle GC across the DAG** (Part III) — the real fault line. The
   arenas+refcount protocol makes it tractable, but its correctness
   (branches/folds/cache/skip/error) is the hard engineering.
2. **Forced Arrow marshalling** for non-threadable controllers (Python GIL-bound,
   **all** R) → Arrow IPC/shared memory. Irreducible but bounded to those
   controllers; the step-cache becomes local to the worker.
3. **Feature-space EXPLAIN** — outputs of feature dimension, opaque to the core
   (stored/transmitted, not interpreted). Minor breach of the visibility principle,
   bounded since non-interpreted.
4. **Tier 1 adaptation of augmentation controllers** (accepting randomness as
   input) — porting cost, not a blocker.

### Open questions

- `describe` blob format: canonical JSON (v1, debuggable) vs msgpack (compact) —
  decide at implementation time.
- `rng_fill` via upcall for feature-shaped draws: chunked Tier 1 or Tier 2
  accepted? (v1: Tier 2.)
- Arena granularity: per fold, per branch, or per variant? (Impacts memory peak
  vs sweep frequency.)
- Streaming / out-of-core (cf. spec §21 Q1): incompatible with the eager
  materialization assumed here; out of scope for v1.

---

## Part VIII — Core language: Rust vs C++ (open decision)

**Status: not settled — to be revisited.** This part records both paths on equal
footing and the criteria that will decide between them. It is not a verdict.

Factual context:
- **nirs4all-io is already in Rust** (the loaders).
- **nirs4all-methods is in C++** (validated, portable).
- In **both** options, the C++ methods **remain C++**, behind the C ABI
  (native controllers). No rewrite of methods is at stake here; the question
  concerns only the **language of the core** (graph, phases, folds, OOF, lineage,
  scheduler, RNG, ABI).

### VIII.1 The reframing specific to this architecture

The classic Rust vs C++ debate is biased here by a design fact: **the architecture
already imposes a C ABI between the core and *all* controllers, native methods
included** (Part II). nirs4all-methods enters as a controller that fills the
`ControllerVTable` with `extern "C"` fns. The core calls C++ methods through the
same vtable as Python controllers: a C ABI call, **zero cost**.

Consequence: the historical pro of C++ ("same language as the methods, so native
integration") is **largely neutralized** — and coupling the core to the methods via
shared C++ types would be an **anti-pattern** (it would break the polyglot symmetry
and the clean ABI boundary). The debate thus reduces to the *intrinsic* qualities
of the language for a **concurrent control plane with cross-FFI liveness**, plus
binding tooling.

### VIII.2 Rust (for the core)

| | |
|---|---|
| **Pros** | • **Memory safety without GC, exactly on blocker #1**: the handle liveness bookkeeping (refcounts, arenas, promotion — Part III) is logic *internal* to the core → in safe Rust, the compiler prevents UAF. A handle UAF = crash in the user's Python/R process, the worst to debug.<br>• **Concurrency without data races**: `Send`/`Sync` protect the scheduler at compile time.<br>• **Binding/Arrow stack = proven template**: pyo3+maturin, extendr, wasm-bindgen/napi, arrow-rs. This is the textbook architecture of Polars, DataFusion, pydantic-core, tokenizers, ruff.<br>• **Cargo**: reproducible build, deps, cross-compilation of wheels without CMake/vcpkg.<br>• **nirs4all-io already in Rust** → ramp-up done, Rust↔C++ bridge already practiced, io ↔ core without a shim. |
| **Cons** | • **Rust↔C++ bridge** for nirs4all-methods: linking the C++ lib (build.rs + `cc`, or `cxx`). Standard, but a build step (already solved on the io side).<br>• The FFI itself remains `unsafe` on both sides — Rust secures the complex bookkeeping, not raw pointer passing.<br>• Younger numeric ecosystem (barely relevant: the core does not compute). |

### VIII.3 C++ (for the core)

| | |
|---|---|
| **Pros** | • **Native integration with nirs4all-methods** (but §VIII.1: gain largely illusory given the C ABI, and coupling would be an anti-pattern).<br>• **Arrow has its reference implementation in C++**; direct Eigen/BLAS/LAPACK.<br>• pybind11 (Python), Rcpp (R) mature; direct ABI/layout control. |
| **Cons** | • **Manual memory safety, directly on blocker #1**: UAF/double-free/handle leaks across FFI = caught at runtime (ASan/UBSan + discipline), not prevented at compilation.<br>• **Manual concurrency**: scheduler data races at the developer's expense.<br>• **Low-level binding/build**: wheels by hand (scikit-build/cibuildwheel), WASM via emscripten, no package manager, CMake dependency hell.<br>• **Inconsistency with nirs4all-io (Rust)**: two systems languages in the same codebase, two toolchains.<br>• No recent template of "C++ core + multi-language bindings + Arrow" as a new project. |

### VIII.4 Resulting topology (identical in both cases)

```
<core language>                        C++                         per language
┌──────────────────────────┐          ┌─────────────────┐        ┌────────────────┐
│ dagml-core (control)      │  C ABI   │ nirs4all-methods│  pyo3  │ sklearn/torch  │
│  graph, folds, OOF,       │◄────────►│  (validated     │◄──────►│ (per language) │
│  lineage, scheduler, RNG  │ extern"C"│   C++ controllers)│extendr│ mlr3 (R)      │
│ nirs4all-io (Rust)        │          └─────────────────┘        └────────────────┘
└──────────────────────────┘
```

The Rust vs C++ choice only changes the left block. nirs4all-io is in Rust; if the
core is in Rust, io ↔ core is shim-free; if the core is in C++, io (Rust) also
becomes a controller/lib behind a C boundary.

### VIII.5 Facts that point a direction, and criteria for later

Three facts orient (without deciding):

1. **nirs4all-io already in Rust** → coherence + ramp-up done → leans Rust.
2. **The hardest blocker is handle liveness + the concurrent scheduler** → the
   project's weak point is Rust's strength → leans Rust.
3. **nirs4all-methods in validated C++** → but neutralized by the C ABI (§VIII.1)
   → barely points either way.

Criteria to evaluate when deciding:

- Does the team commit to Rust **beyond io** (including the core)? (io in Rust
  suggests yes.)
- Does the control plane become a bottleneck at scale (millions of
  lineage/prediction records) → favors a compact compiled core (Rust or C++).
- Actual experienced cost of the Rust↔C++ bridge for the methods (cf. `extern "C"`
  shim, to be prototyped).
- Serious WASM/JS need → clearly favors Rust (wasm-bindgen).

### VIII.6 Sub-question: language for new methods

Independent of the core language, but related:

- **Existing** methods (validated) → stay **C++**, behind the ABI. No re-validation.
- **New** methods → if the core is Rust, writing them in Rust lets them live in the
  native core **without even a shim**; otherwise C++. The ABI boundary makes both
  interchangeable, so no urgency to standardize.
