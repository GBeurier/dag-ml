# Native Persistence Layer — architecture impact & merit report

> **Scope.** Evaluates a *proposed new project*: a native (Rust), fast, light-dependency library
> sitting "between `nirs4all-io` and `dag-ml-data`" that owns **all persistence of predictions and
> scores** (save/load prediction tables + metric/score records) so this persistence is available
> cross-language (Python/R/MATLAB/WASM) without heavy Python deps.
>
> **Method.** Grounded in the actual code of all four candidate layers (cited `file:line`). No
> speculation. Aligns to the Migration Mandate (`dag-ml/AGENTS.md:12-26`) and the war-room target
> split (`TARGET_RESPONSIBILITY_SPLIT.md`).
>
> **Assembled 2026-06-23.**

---

## Executive summary (read this first)

1. **dag-ml already persists prediction tables natively in Rust, light-dep.** `FilePredictionCacheStore` (`dag-ml-core/src/runtime.rs:1113`) writes per-fold + OOF + aggregated (`avg`/`w_avg`) prediction blocks to a JSON directory store with sha256 fingerprint validation, plus an `export-prediction-cache-store` / `validate-prediction-cache-store` CLI pair (`dag-ml-cli/src/main.rs:1836-1862`). Its only deps are `serde_json` + `sha2`. **The "predictions" half of the proposal is ~80% already built.**

2. **The real gap is SCORES, not predictions.** dag-ml *computes* full per-fold/per-target/multi-metric reports (`RegressionMetricReport`, `metrics.rs:100`) but **never persists them** — they collapse to a winning scalar + rank list inside the bundle (`SelectionDecision`, `selection.rs:254`). There is no queryable metrics table. That, plus all-partition (test/holdout/final) persistence and a columnar/queryable surface, is the genuine missing capability.

3. **The proposed *placement* is wrong.** The `nirs4all-io ↔ dag-ml-data` seam is the **input** path (io emits a `CoordinatorDataPlanEnvelope` — schema + plan + identity, no outcomes). Predictions/scores are **outputs** and never flow through it. io explicitly disclaims storage (`nirs4all-io/CLAUDE.md:156-158`); dag-ml-data explicitly disclaims "OOF prediction blocks" and model outputs (`dag-ml-data/docs/ARCHITECTURE.md:53-58`). Output persistence belongs on the **dag-ml side**, downstream of the model run.

4. **A separate repo is not justified.** The capability is generic ML-coordination output — exactly dag-ml's charter (`AGENTS.md:17` lists "prediction/score persistence" as a dag-ml mandate). A new repo would duplicate `bundle.rs`/`runtime.rs`/`provenance.rs`, add a release-train hop (ADR-10), fork the fingerprint/schema contract, and split a single replay story across two codebases. **Recommendation: extend dag-ml's existing prediction-cache into a queryable predictions+scores store, in a dag-ml-core submodule (or a thin `dag-ml-store` crate inside the dag-ml workspace).** No new project.

5. **Back-compat is a one-way bridge, not a shared format.** The frozen 0.9.x nirs4all workspace (SQLite + Parquet + joblib) and `.n4a` (`nirs4all/CLAUDE.md:5`) must stay as-is. The native store does **not** read/write that schema; nirs4all keeps a thin Python writer that mirrors native records into the legacy workspace during the migration window (ADR-02 bundle-readability SLA). The joblib/pickle artifact format is the hard cross-language wall and is **out of scope** for a predictions/scores store regardless.

---

## 1. Current-state map — who persists predictions/scores today

Two independent, overlapping persistence systems exist.

### 1a. nirs4all (Python, the production system) — `nirs4all/pipeline/storage/`

A hybrid **SQLite (metadata + scalar scores) + Parquet (dense arrays) + content-addressed joblib (fitted artifacts)** workspace, schema `SCHEMA_VERSION = 2` (`store_schema.py:28`).

| What | Where | Format | Evidence |
|---|---|---|---|
| Runs / pipelines / chains / artifacts / logs / projects | SQLite tables | rows + JSON-in-TEXT | `store_schema.py:44-186` (7 tables) |
| Prediction rows (per-fold `fold_0..`, ensemble `avg`, refit `final`) + scalar scores | `predictions` table | SQLite, upsert on natural key | `store_schema.py:111-148`; fold/refit semantics `workspace_store.py:1216-1238` |
| CV score summaries (`cv_val_score`, `cv_scores` JSON), refit + aggregated-ensemble scores | `chains` table | SQLite columns | `store_schema.py:77-109` |
| Dense arrays: `y_true`, `y_pred`, `y_proba`(+shape), `sample_indices`, `weights`, per-sample metadata (JSON) | `arrays/<dataset>.parquet` | Parquet (pyarrow write / polars read), Zstd-3 | `array_store.py:124-140`, `:292-297`, `:397-399` |
| Fitted models | `artifacts/ab/<sha256>.joblib` | **joblib / pickle**, content-addressed, ref-counted | `workspace_store.py:190-210`, `:1353-1379` |
| Portable export | `exports/*.n4a` (ZIP) | JSON manifest/trace/chain + joblib artifacts | `workspace_store.py:2271-2383`; richer `bundle/generator.py:305-378` |

- **WAL mode** is enabled (`store_schema.py:606-607`), with an `RLock` + retry-on-lock decorator (`workspace_store.py:116-157`) and POSIX/Windows file locking on the Parquet dir (`array_store.py:202-243`).
- **Deps (persistence-relevant):** `sqlite3` (stdlib), `pyarrow>=14` (Parquet write — heavy), `polars>=1.0` (Parquet read + all DataFrame returns — heavy Rust binary), `numpy>=2`, `joblib>=1.2` (Python-only), `pandas` (convenience only), `pyyaml`. The **`Predictions` facade** (`data/predictions.py:183`) buffers rows during a run and flushes via `WorkspaceStore.save_prediction` (row-by-row SQLite) + `array_store.save_batch` (one Parquet write per dataset) inside a transaction (`predictions.py:927-1044`). Store is injected via `register_store_backend` (`predictions.py:496-514`) — data layer never imports pipeline layer.

### 1b. dag-ml (Rust, the migration target) — `dag-ml-core/`

A **JSON-only, fingerprint-validated, replay-grade** persistence layer. All persisted types derive serde.

| What | Type (file:line) | Persisted? |
|---|---|---|
| Per-fold / OOF sample-level predictions | `PredictionBlock` (`oof.rs:29-40`) | ✅ in cache store |
| Aggregated avg / w_avg / median predictions | `AggregatedPredictionBlock` (`aggregation.rs:67-78`); methods `policy.rs:91-100` | ✅ in cache store |
| `y_true` target block | `RegressionTargetBlock` (`metrics.rs:39-46`) | serde-able |
| Prediction-cache payload (values) + manifest (metadata) | `BundlePredictionCachePayloadSet` (`bundle.rs:753`), `BundlePredictionCacheRecord` (`bundle.rs:319`) | ✅ JSON files |
| Top-level run manifest | `ExecutionBundle` (`bundle.rs:914-938`) — schema-versioned (`bundle.rs:121`, `SchemaMigrationPolicy bundle.rs:41`) | ✅ `execution_bundle.json` |
| Selection outcome (winning candidate + ranks) | `SelectionDecision` / `RankedCandidate` (`selection.rs:254`, `:248`) | ✅ in bundle |
| **Full per-fold/per-target/multi-metric scores** | `RegressionMetricReport` (`metrics.rs:100-113`) | ❌ **computed, never persisted** |
| Lineage / provenance | `ResearchProvenancePackage` (`provenance.rs:35`), W3C PROV-JSONLD + RO-Crate + OpenLineage (`provenance.rs:17-26`, `:294`, `:396+`) | ✅ JSON files |

- **The prediction-cache store** is `FilePredictionCacheStore` (`runtime.rs:1113`): a directory of `prediction-cache-<hex>.json` payloads + a `prediction_cache_manifest.json` (`runtime.rs:939`, `:1024-1031`), written by `write_payload_set` (`runtime.rs:1121`) and re-validated against the bundle (fingerprint match) by `validate_file_prediction_cache_store`. CLI: `export-prediction-cache-store` (`main.rs:1836-1853`), `validate-prediction-cache-store` (`main.rs:1854-1862`). It is consumed via the `RuntimePredictionCacheStore` trait (`runtime.rs:925`) with File / Columnar / InMemory impls.
- **Deps:** `dag-ml-core` = `serde`, `serde_json`, `yaml_serde`, `indexmap`, `sha2`, `thiserror`, `tracing`. **No arrow, no parquet, no sqlite, no DB.** The columnar f64 representation (`ColumnarPredictionCacheStore`, `runtime.rs:1724`; `layout: "column_major_f64"`, `capi/src/lib.rs:3537`) is an **in-memory ABI hand-off only — never serialized to disk.**

### 1c. Quantified overlap (nirs4all SQLite+Parquet ⟷ dag-ml bundle/cache)

| Capability | nirs4all (Python) | dag-ml (Rust) | Overlap |
|---|:--:|:--:|:--:|
| Per-fold predictions persisted | ✅ Parquet+SQLite | ✅ `PredictionBlock` JSON | **Full** |
| OOF predictions persisted | ✅ (query over rows) | ✅ cache store (Validation-only) | **Full** |
| Ensemble avg / w_avg persisted | ✅ `avg` rows + `final_agg_*` | ✅ `AggregatedPredictionBlock` | **Full** |
| Final refit predictions persisted | ✅ `final` rows | ⚠️ representable, but **cache `validate()` forces `partition==Validation`** (`bundle.rs:353`) | **Partial** |
| Test/holdout predictions persisted | ✅ `partition` column | ⚠️ block type exists; cache store rejects non-validation | **Partial** |
| Full metric/score records (per fold/target/metric) | ✅ `scores` JSON + `cv_scores` | ❌ collapsed to scalar+rank | **None — gap** |
| Manifest / run metadata | ✅ `runs`/`pipelines`/`chains` | ✅ `ExecutionBundle` | Conceptual |
| Lineage / provenance / PROV / RO-Crate | ❌ (trace.json only) | ✅ full | dag-ml only |
| Fitted-model artifacts | ✅ joblib (Python-only) | handles/refs only (`RefitArtifactRecord bundle.rs:846`) | Different concern |
| Queryable analytical surface ("RMSE for target X across folds") | ✅ SQL / polars | ❌ load-whole-payload-by-key | **None — gap** |

**Net:** prediction-table persistence is **substantially duplicated already**. The non-overlapping, genuinely-missing pieces are: **(a) a persisted metric/score table, (b) all-partition (not OOF-only) persistence, (c) a columnar-on-disk + queryable surface.**

---

## 2. The gap — what actually blocks cross-language persistence

The cross-language problem is **not** "predictions can't be persisted natively" — dag-ml already does that with `serde_json` + `sha2`. The real blockers are narrower and concrete:

1. **The *production* persistence is Python-only by construction.** Every read/write in nirs4all goes through `sqlite3` (stdlib API), `pyarrow` (Parquet write), `polars` (Parquet read, and every `query_*` returns a `polars.DataFrame` — `store_protocol.py:20`, `workspace_store.py:394`). The *file formats* (SQLite, Parquet, JSON) are language-neutral, but the *access layer* must be reimplemented per language. (Evidence: §1a deps; `predictions.py:927-1044` flush path.)

2. **Fitted artifacts are joblib/pickle** (`workspace_store.py:190-210`) — fundamentally Python-object serialization. R/MATLAB/WASM can read the bytes but cannot deserialize the models. **This is the true cross-language wall, and it is *orthogonal* to a predictions/scores store** — predictions and scores are plain numbers; models are the unportable part. A predictions/scores persistence layer does *not* solve, and need not touch, the artifact problem.

3. **dag-ml persists for *replay/lineage*, not for *analytical query*.** The cache store is keyed by `requirement_key`; you load a whole payload, you cannot query across folds/targets (`runtime.rs:1113-1230`). And the **scores never get persisted at all** (`RegressionMetricReport` → `CandidateScore` → discarded except winner; `metrics.rs:154`, `selection.rs:248`). So even on the native side, there is no queryable predictions+scores artifact.

**Conclusion of the gap analysis:** the bottleneck is *not* "no native persistence." It is (i) the production path is Python-coupled, and (ii) the native path persists predictions-for-replay but **not scores, not all partitions, and not in a queryable form.** A focused effort closes (ii) and lets every binding inherit it via the ABI; (i) dissolves automatically once the native store is the source of truth.

---

## 3. Proposed-layer design (evaluated against the existing code)

### 3a. Scope — what it should own

| Candidate scope | Verdict | Why |
|---|---|---|
| Save/load prediction tables (per-fold/OOF/agg/final/test) | **In** (mostly exists) | `PredictionBlock` / `AggregatedPredictionBlock` already serialize; needs the all-partition relaxation (lift `bundle.rs:353` Validation-only constraint for a results store) |
| Save/load metric/score records (per fold/target/metric) | **In** (the real new work) | `RegressionMetricReport` (`metrics.rs:100`) is serde-ready but unpersisted; add a `MetricRecord` store |
| Queryable/indexed read surface | **In** | the missing analytical capability |
| Lineage / provenance | **Out** — already dag-ml's `provenance.rs` | don't fork it |
| Fitted-model artifacts | **Out** | joblib/pickle/ONNX is a separate, harder problem; predictions/scores are numbers |
| Dataset/feature persistence | **Out** | dag-ml-data owns `.n4d` feature buffers (`buffer_file_store.rs:1-9`); not this layer |

Scope should be **predictions + scores ONLY**, exactly as the proposal states. That is correct and matches the gap.

### 3b. Placement — three options, ranked

| Option | Fit to charters | Cost | Verdict |
|---|---|---|---|
| **A. Extend dag-ml's prediction-cache into a predictions+scores store** (submodule in `dag-ml-core`, or a thin `dag-ml-store` crate inside the dag-ml workspace) | ✅ `AGENTS.md:17` lists "prediction/score persistence" as a **dag-ml mandate**; reuses `bundle.rs`/`runtime.rs`/fingerprints; one replay story | Lowest — incremental | **RECOMMENDED** |
| B. New crate inside **dag-ml-data** | ❌ Violates charter: dag-ml-data "must not expose OOF prediction blocks" (`docs/ARCHITECTURE.md:53-58`), "must not own … model execution" (`CLAUDE.md:11`); predictions/scores are outputs | Medium + boundary breach | Reject |
| C. **Standalone new repo** "between nirs4all-io and dag-ml-data" | ❌ Wrong seam (io→dag-ml-data is the *input* path; predictions/scores never traverse it — `nirs4all-io/CLAUDE.md:156-158`, io emits only a `CoordinatorDataPlanEnvelope`); duplicates dag-ml persistence; +1 release-train hop (ADR-10); forks the fingerprint/schema contract | Highest — new CI, license, repo, cross-repo contract sync | Reject |

The proposal's stated placement ("between nirs4all-io and dag-ml-data") is **architecturally mis-located**: that seam carries the *input* `CoordinatorDataPlanEnvelope` (schema + plan + sample/target/group/origin identity, **no outcomes** — confirmed: the envelope schema has no prediction/score/metric field). Output persistence sits **downstream of the model run**, on the dag-ml side.

### 3c. Storage tech for "fast + light deps"

dag-ml's hard constraint is **no host runtime deps in the core** (`dag-ml/CLAUDE.md` crate-direction note: `dag-ml-core` = "no host runtime deps"). Compare:

| Tech | Deps added | Cross-lang read | Query | Fit to dag-ml |
|---|---|---|---|---|
| **serde_json (today's cache store)** | none (already in) | trivial | none | ✅ current baseline |
| **Arrow IPC** | `arrow-array`/`arrow-ipc`/`arrow-schema` (~1 family) | excellent (R/MATLAB/WASM/Python all read Arrow) | columnar scan | ✅ already precedented & **isolated** in `dag-ml-data-arrow` behind a non-default feature (`dag-ml-data-arrow/Cargo.toml:13-16`) |
| Parquet-rs | `parquet` + arrow | excellent + compression | predicate pushdown | ⚠️ heavier; matches nirs4all's Parquet so easy bridge |
| redb / sqlite-rs | embedded DB | sqlite=universal; redb=Rust-only | indexed | ⚠️ adds a DB dep to a JSON-pure core |
| custom columnar | none | must re-implement readers everywhere | custom | ❌ reinvents Arrow |

**Recommendation:** keep `serde_json` as the canonical/replay format (zero new deps, already validated), and add **Arrow IPC** as the *columnar/queryable* representation — gated behind an optional feature, mirroring the existing `dag-ml-data-arrow` pattern and the in-memory `ColumnarPredictionCacheStore` (`runtime.rs:1724`) that already produces column-major f64. Arrow gives R/MATLAB/WASM native read with one isolated dependency family and aligns with nirs4all's Parquet (Parquet = Arrow-on-disk), easing the legacy bridge. Avoid adding an embedded DB to `dag-ml-core`.

### 3d. Cross-language binding surface

Reuse the **proven ecosystem pattern** (one Rust core + C ABI + thin per-language wrappers) already shipped by `nirs4all-io` (Python/R/MATLAB/WASM/C — `nirs4all-io/COMPAT.md`) and `dag-ml-capi`. The store surface crosses the ABI as: canonical JSON descriptors + dense f64 tensors (released via `dagml_f64_columnar_tensor_free`, `capi/src/lib.rs`) — **predictions/`y_true`/scores as data, never as host handles** (matches `AGENTS.md:32-36`). No new binding mechanism is needed.

### 3e. Relationship to dag-ml's existing bundle/prediction-cache

**Layer on top / extend — do not merge-away or supersede.** The `ExecutionBundle` + `FilePredictionCacheStore` are the *replay manifest + OOF cache* and must stay (replay/lineage depend on them). The new capability is additive:

1. Relax the Validation-only constraint (`bundle.rs:353`) — or add a sibling `ResultsStore` — so test/holdout/final partitions persist.
2. Add a persisted `MetricRecord` table (from the already-computed `RegressionMetricReport`, `metrics.rs:100`) — the one genuinely new artifact.
3. Add the Arrow/columnar on-disk write + an index for analytical queries.

---

## 4. Impact

### 4a. On the frozen 0.9.x nirs4all contract (`nirs4all/CLAUDE.md:5`)

The workspace SQLite/Parquet schemas, run-manifest layout, and `.n4a` format are **stable within 0.9.x**. The native store **must not** attempt to be that format. Migration path (ADR-02 schema-evolution SLA + ADR-17 cutover): native store is the source of truth; nirs4all keeps a **thin Python shim** that mirrors native predictions/scores into the legacy SQLite+Parquet workspace during the dual-run window, so Studio and existing `.n4a` consumers see no change. The joblib artifact wall is untouched (out of scope). This is a *bridge*, consistent with ADR-14's managed-debt exception — but note the conflict with nirs4all's no-shims rule (war-room open decision #3) is the maintainer's call.

### 4b. On dag-ml's `bundle.rs`

Additive: lift one validation constraint, add a metric-record store, add an optional Arrow writer. ~No churn to the existing serde types or the replay path. Consolidation *toward* dag-ml, not a fork.

### 4c. On the migration timeline

Option A rides the **existing** ADR-10 release train (dag-ml-data → dag-ml → nirs4all). Option C adds a *fourth* package to that train + a new cross-repo contract to keep JSON-identical — directly contrary to the war-room's "fewer moving parts" posture and the "don't reinvent dag-ml" warning (Tier-3 trap, war-room README:86-95).

### 4d. On performance

Today's cache store is `serde_json` (text); Arrow/columnar would be **faster** than both JSON and the nirs4all read-modify-write-whole-Parquet append (`array_store.py:350-357`). But note the war-room caveat: **dag-ml perf is only sanity-probed, not benchmarked** (war-room README:129). Any "fast" claim must be benchmarked against the SQLite+Parquet baseline before it is asserted.

### 4e. On dependency weight

Option A in `dag-ml-core` keeps the core pure (serde_json + sha2); Arrow is opt-in and isolated (precedent: `dag-ml-data-arrow`). A standalone repo would re-pull serde/sha2/arrow and re-implement the fingerprint contract — *more* total dependency surface across the ecosystem, not less.

### 4f. On maintenance / another-repo cost

A new repo = new CI green-gate, license headers (the dual-CeCILL/AGPL + commercial policy), `validate_contracts.py` cross-repo parity job, release mechanism, RTD docs, and a contract that must stay byte-identical with dag-ml's prediction-cache schemas. This is the single largest hidden cost and the strongest argument against Option C.

---

## 5. Merit / bien-fondé — recommendation

**The capability is justified; a separate project is not.**

- The *cross-language predictions+scores persistence* goal is real and aligns with the Migration Mandate, which **explicitly assigns "prediction/score persistence" to dag-ml** (`AGENTS.md:17`) so "every binding (Python/R/MATLAB/WASM) gets it for free" (`AGENTS.md:20-21`).
- But ~80% of it (prediction tables, manifests, fingerprints, replay) **already exists** in `bundle.rs` + `runtime.rs` + `provenance.rs` with the exact light-dep profile the proposal wants (`serde_json` + `sha2`). A new repo would duplicate this.
- The genuinely-new work — **persisted score/metric records, all-partition predictions, a columnar/queryable surface** — is small, additive, and lands cleanly *inside dag-ml*.
- The proposed *placement* (io ↔ dag-ml-data) is on the **input** seam; predictions/scores are **outputs** and belong downstream of the model run. Both neighbors' charters explicitly disclaim this (`nirs4all-io/CLAUDE.md:156-158`; `dag-ml-data/docs/ARCHITECTURE.md:53-58`).

### Recommendation (ranked)

1. **DO (recommended): Extend dag-ml's prediction-cache into a predictions+scores store**, as a submodule of `dag-ml-core` or a thin `dag-ml-store` crate *inside the dag-ml workspace*. Add (a) a persisted `MetricRecord` from the existing `RegressionMetricReport`, (b) all-partition support (relax `bundle.rs:353`), (c) an optional Arrow-IPC columnar/queryable representation (feature-gated, like `dag-ml-data-arrow`). Expose via the existing C ABI; bind per-language with the io/dag-ml pattern. nirs4all keeps a thin legacy-workspace mirror shim for the 0.9.x contract.
2. **Acceptable alternative:** same code as a separate *crate within the dag-ml workspace* (not a separate repo) if the maintainer wants clearer module boundaries — keeps one repo, one release-train slot, one contract.
3. **Reject:** a crate inside `dag-ml-data` (charter breach) or a standalone new repo "between io and dag-ml-data" (wrong seam, duplication, +1 release/contract/CI hop).

---

## 6. Risks & open questions for the maintainer

1. **Scores schema.** No score/metric *persistence* schema exists yet (only `node_result.schema.json` has an optional `metrics` map). A new `metric_record.schema.json` must be designed and added to the conformance pack — decide its grain (per fold × target × metric).
2. **Validation-only constraint.** Lifting `bundle.rs:353` (cache = Validation-only) for a results store risks blurring the OOF-cache (a leakage-safety boundary) with a general results store. Keep them as *distinct* stores, or gate the relaxation carefully — do not weaken the OOF invariant (`dag-ml/CLAUDE.md` engineering rules).
3. **Arrow vs JSON canonical form.** Two on-disk forms (JSON for replay/fingerprint + Arrow for query) risk drift. Decide which is canonical and whether the Arrow form is derived-and-disposable or first-class.
4. **Perf is unmeasured.** The "fast" claim is unproven (war-room README:129). Benchmark against SQLite+Parquet before asserting it.
5. **Back-compat posture.** The legacy-workspace mirror shim conflicts with nirs4all's absolute no-shims rule (war-room open decision #3) — only the maintainer can sanction the ADR-14 managed-debt exception.
6. **Artifacts stay out.** Confirm the layer is predictions/scores *only*; the joblib→neutral-model-format problem (ONNX / nirs4all-methods / dag-ml handles) is separate and must not be smuggled in.
7. **Reducer-vocabulary drift.** `robust_mean`/`exclude_outliers` exist data-side only (war-room README:132); a scores/agg store must reconcile reducer names on the release train.

---

*Provenance: grounded in `nirs4all/pipeline/storage/*`, `nirs4all/data/predictions.py`, `dag-ml-core/src/{bundle,runtime,aggregation,metrics,oof,selection,provenance,policy}.rs`, `dag-ml-cli/src/main.rs`, `dag-ml-data/{CLAUDE.md,docs/ARCHITECTURE.md,crates}`, `nirs4all-io/{CLAUDE.md,COMPAT.md,crates}`. All `file:line` cited inline. 2026-06-23.*
