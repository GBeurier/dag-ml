# nirs4all-core → dag-ml migration — war room

> **Status:** groundwork / prep phase (no code migration started). **Production nirs4all is
> untouched and stays untouched.** This folder is the consolidated study + plan so it never
> gets lost again. Assembled 2026-06-23 from a full multi-agent recon of every migration doc
> across the ecosystem (15 clusters, 83 docs classified).
>
> **Goal of the chantier:** make **dag-ml** (Rust + C-ABI reproducible DAG coordinator) the
> real execution core of the production Python library **nirs4all**, *without* breaking the
> frozen 0.9.x/0.10.x public surface and *without* disturbing the production `main`.

This is an **index + synthesis**, not a copy dump. Every source doc is referenced at its
canonical home (cross-repo docs use `repo:path` notation, dag-ml docs use relative links) so
nothing drifts. Companion proposals live next to this file:

- [`WORKING_STRATEGY.md`](WORKING_STRATEGY.md) — how to develop the migration while keeping prod alive (repo / branch / worktree / backend-selector). *Answers ask #3.*
- [`PARITY_AND_PERF_HARNESS.md`](PARITY_AND_PERF_HARNESS.md) — how to guarantee non-regression (parity) and measure performance, automated. *Answers ask #4.*

---

## TL;DR readiness verdict

**dag-ml 0.2.0 is ready to back the nirs4all _control plane_, not yet to run production
_numerics_.** It is a closed, CI-gated, contract-frozen Rust coordinator built for exactly
this problem (reproducible, OOF/leakage-safe, identity-keyed DAG coordination). It already
implements every *coordination* capability the migration needs. By deliberate design it
**never touches feature matrices** — all NIRS numerics must be supplied by nirs4all as host
controllers, and today the only production host adapter is **sklearn**.

The migration is therefore **mostly a nirs4all-side decoupling + bridging job**, not a
"finish dag-ml" job.

---

## The map — where the prepared study actually lives

Read these three tiers in order. **Tier 3 is a trap: it is parallel design, not the plan.**

### Tier 1 — Authoritative migration design (dag-ml side) — *plan against these*

| Doc | Rel. | Status | Why it matters |
|---|---|---|---|
| [`../COORDINATOR_SPEC.md`](../COORDINATOR_SPEC.md) | ★★★ | current | Normative product contract; has an explicit **"Confrontation With Current nirs4all Pipeline" + migration map**. The alignment source of truth. |
| [`../CAPABILITY_MATRIX.md`](../CAPABILITY_MATRIX.md) | ★★★ | current | States outright the goal **is to replace the nirs4all core engine**; per-feature responsibility map + 4-stage MVP→replacement path. |
| `../design/DSL_NIRS4ALL_PARITY.md` | ★★★ | local design-source | Live acceptance criterion: maps **every nirs4all construct → dag-ml NodeKind**, importer status, gaps/regression list. |
| `../STATUS.md` | ★★★ | local-only current* | Authoritative ledger of what's implemented in 0.2.0 vs backlog. |
| [`../SUPPORTED.md`](../SUPPORTED.md) | ★★★ | current | Per-area Supported / Conformance / Experimental / Backlog — what you may depend on as production. |
| `../HOST_ADAPTER_BACKLOG.md` | ★★★ | local-only current* | Defines the **process-adapter JSONL wire protocol** nirs4all must implement; confirms sklearn adapter shipped. |
| `../MVP_ACCEPTANCE.md` | ★★★ | local-only current* | dag-ml ↔ dag-ml-data ownership boundary + UC6/UC11 acceptance the MVP must satisfy. |
| `../TEST_PLAN.md` | ★★★ | local-only current* | Most complete inventory of what dag-ml validates today; the ledger to diff nirs4all behaviour against. |
| [`../ABI.md`](../ABI.md) | ★★★ | current | Full C-ABI surface (vtables, ownership, Arrow boundaries) — the in-process/WASM host path. |
| `../ROADMAP.md` / `../FINAL_RELEASE_AUDIT.md` | ★★ | local-only current* | Phase status + 0.2.0 release verdict + green-gate command sequence. |
| `../HETEROGENEOUS_MULTISOURCE_REPETITIONS_ROADMAP.md` | ★★★ | local-only current* | Phased D0–D10 roadmap for the NIRS-critical shared-target-multi-spectra feature. |
| [`../PERFORMANCE.md`](../PERFORMANCE.md) | ★★ | current | **Perf is only sanity-probed, not benchmarked** — a named cutover risk (see harness doc). |
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) / [`../AGGREGATION_INTEROP.md`](../AGGREGATION_INTEROP.md) / [`../OOF_FIXTURES.md`](../OOF_FIXTURES.md) / `../STUDIO_LITE_WASM_GAPS.md` | ★★ | mixed | Crate map, reducer interop, canonical OOF fixtures, remaining execution gaps. `STUDIO_LITE_WASM_GAPS.md` is local-only. |

**ADRs — framed as Phase-0 of the nirs4all integration** ([`../adr/README.md`](../adr/README.md)):

| ADR | Role in the migration |
|---|---|
| [ADR-17 cutover-rollback](../adr/ADR-17-cutover-rollback.md) | **The migration-strategy ADR**: backend selector (`legacy` \| `dag-ml` \| `dual`), dual-run diff within tolerance, zero-cost rollback. |
| [ADR-01 compatibility-ledger](../adr/ADR-01-compatibility-ledger.md) | Defines **"no regression"** = the compatibility ledger + per-model-class numeric tolerance table the dual-run must satisfy. |
| [ADR-02 schema-evolution-sla](../adr/ADR-02-schema-evolution-sla.md) | Additive-then-promote SLA + bundle-readability guarantee → never orphan existing `.n4a`. |
| [ADR-14 deprecation-policy](../adr/ADR-14-deprecation-policy.md) | **Managed-debt exception** legitimizing legacy-path / dual-read shims during the window (⚠ conflicts with nirs4all's no-shims rule — see open decisions). |
| [ADR-10 release-train](../adr/ADR-10-release-train.md) | Scripted dag-ml-data → dag-ml → nirs4all release ordering the rollout rides. |
| [ADR-05 repetition-cv-invariant](../adr/ADR-05-repetition-cv-invariant.md) · [ADR-11 error-taxonomy](../adr/ADR-11-error-taxonomy.md) | Leakage invariant the bridge can't drop · typed error substrate. |
| ADR-03/04/06/07/08/13/15/16/19 | Per-feature semantics the bridge must reproduce (branches, tag/exclude masks, signal-type, reducers, sessions, process-adapter security, GIL/async, artifact security, multisource units). |

**Archived design-source** (historical but richest code-level orders) — `../design/source/_archive/`:
`dag_ml_specification_v1.md` **§19 = 10-step migration plan**; `dag_ml_externalization_from_code.md` = **9-step extraction order** from the live nirs4all engine; `dag_ml_use_cases.md` = **Annexe-A DSL→NodeKind acceptance table**.

> *`current*` = the doc is part of the "kept-locally / untracked" set in `dag-ml/.gitignore`
> (lines 23–32). It exists on disk here but is **not committed** to dag-ml. See open decision #7.

### Tier 2 — nirs4all side: what we migrate *from* + the executable gate

| Doc (`repo:path`) | Rel. | Why it matters |
|---|---|---|
| `nirs4all:docs/_internal/specifications/heterogeneous_multisource_repetitions.md` | ★★★ | **The single most migration-relevant nirs4all-side doc**: reviewed cross-repo co-design with shared contracts, ownership split, phased N0–N8 / D0–D6 roadmaps. |
| `nirs4all:docs/_internal/god_classes_modularization.md` | ★★★ | Decomposition backlog for the **12 >2k-line god classes** (orchestrator / merge / branch / base_model / workspace_store / predictions / dataset) that own OOF/refit/branch-merge/storage — **prerequisite to crossing the ABI**. |
| `nirs4all:docs/_internal/prediction_to_pipeline.md` | ★★★ | Existing replay / chain / expanded_config / trace / bundle round-trip + its gaps — maps onto dag-ml's replay/lineage mandate. |
| `nirs4all:docs/source/developer/architecture.md` | ★★★ | Canonical description of the **production core being migrated FROM**: Orchestrator→Executor→StepRunner→controllers over a mutable `SpectroDataset`. |
| `nirs4all:docs/_internal/lib_ML/NIRS4ALL_porting.md` | ★★ | File-by-file generic-vs-NIRS boundary map; names the **`SpectroDataset.x()` / wavelength-injection hotspots** any core swap must solve. (Targets "pipeforge", not dag-ml — boundary input only.) |
| `nirs4all:tests/integration/parity/` (README + `_registry.py` + `cases_*.py` + `test_parity_smoke.py`) | ★★★ | **The executable gate** — see [`PARITY_AND_PERF_HARNESS.md`](PARITY_AND_PERF_HARNESS.md). ~35 frozen cases; explicitly "the contract the future dag-ml backend must reproduce". |

### Tier 3 — Parallel designs — donor/parity input only, **NOT the plan**

These never reference dag-ml; they independently reinvent large parts of it. Treat as a
**requirements / parity inventory** ("what must be preserved"), never as the architecture to build.

| Doc (`repo:path`) | What to harvest | Why it's not the plan |
|---|---|---|
| `nirs4all:docs/_internal/nirs4all_v2_design/00..05 + virtual_data_management_design.md` | Feature-Preservation Matrix, identity schema, node taxonomy, leakage edge cases, numerical-equivalence harness (r²±0.001), public-surface checklist | A Dec-2025 **pure-Python** ground-up rewrite. Keys folds by **row position** (dag-ml rejects this). Internally self-contradictory on backward-compat. |
| `nirs4all:docs/_internal/lib_ML/ML_lib_design.md` | Generic-core stays/moves boundary map | A *third* parallel design ("pipeforge", generic Python core). |

### Cross-repo integration contracts (adjacent, not core)

`dag-ml-data:docs/ADR-0001-nirs4all-connector-ownership.md` · `nirs4all-methods:docs/nirs4all_integration_map.md` (per-class PLS/model → libn4m swap; largely blocked today) · `nirs4all-lite:docs/PARITY.md` (full-Python nirs4all = oracle of record) · `nirs4all-formats:docs/INTEGRATION_NIRS4ALL.md` (leaf readers, no dag-ml content).

---

## Suggested reading order (first day on the chantier)

1. `../CAPABILITY_MATRIX.md` — *what* dag-ml takes over, feature by feature.
2. `../COORDINATOR_SPEC.md` (the nirs4all confrontation section) — *how* it maps.
3. `../design/DSL_NIRS4ALL_PARITY.md` — *exactly* which DSL constructs are covered / gapped.
4. `../adr/ADR-17-cutover-rollback.md` + `../adr/ADR-01-compatibility-ledger.md` — the safety strategy.
5. `../STATUS.md` + `../SUPPORTED.md` + `../HOST_ADAPTER_BACKLOG.md` — what's real today + the wire protocol.
6. `nirs4all:tests/integration/parity/README.md` + `_registry.py` — the gate.
7. `nirs4all:docs/_internal/god_classes_modularization.md` — the decoupling work.
8. `../design/source/_archive/dag_ml_externalization_from_code.md` §(9-step) + `dag_ml_specification_v1.md` §19 — the detailed extraction order.

---

## dag-ml readiness (the honest ledger)

**Ready (control plane):** full COMPILE→PLAN→FIT_CV→SELECT→REFIT→PREDICT→EXPLAIN with
deterministic scheduling; nirs4all-compatible **JSON DSL importer** (list/dict pipelines,
`_or_`/`_cartesian_`/`_chain_`/`_grid_`/`_range_`/`_log_range_`/`_zip_`/`_sample_`, branch/merge/split/sources);
identity-keyed FoldSet + OOF-join-by-sample-id + **leakage refusal by default** (UC6/UC11 as
conformance fixtures); deterministic variant generation/selection with fingerprints; replay
bundles + provenance (RO-Crate / PROV / OpenLineage); C-ABI + process-adapter JSONL path with
a **shipped sklearn adapter**; heterogeneous multisource conformance pack (D8/D9).

**Gaps blocking core migration:**
1. **No Python-object/YAML frontend** — the Rust importer eats serialized JSON; nirs4all must serialize *live* pipeline objects / splitters / sklearn instances to portable descriptors (today `operator_class` is a short name, lossy for live instances).
2. **No production host controllers beyond sklearn** — all 42 nirs4all model estimators currently route through process adapters; the nirs4all-methods/libn4m swap is a separate, largely-blocked effort.
3. **Production data-provider path incomplete** — multi-node graphs need per-node dag-ml-data provider wiring; host-filtered `branch_view` modes (by_metadata/by_tag/by_filter) not yet production.
4. **Performance unmeasured** — only two ignored ~1.5 s sanity probes; no throughput/memory/end-to-end campaign benchmarks; per-task JSONL process-adapter overhead is the biggest unknown at nirs4all scale.
5. **nirs4all-side decoupling not done** — `SpectroDataset.x()` + `TransformerMixinController` wavelength injection hotspots + the 12 god classes must be split along the dag-ml ownership seam.
6. **Parity oracle is a scaffold** — no captured gold baseline, tolerances recorded but unenforced, no dag-ml backend hook wired in yet.
7. **Cross-repo reducer-vocabulary drift** — `robust_mean`/`exclude_outliers` exist data-side only; must reconcile on the ADR-10 release train.

---

## Open decisions — only the maintainer can make these (see proposals for recommendations)

1. **Destination** — dag-ml directly as the new core (recommended; per CAPABILITY_MATRIX/COORDINATOR_SPEC) **vs** the intermediate generic "pipeforge" Python lib **vs** the v2_design pure-Python rewrite. *Pick one; retire the other two (or mark parity-input-only) before any decomposition.*
2. **Mechanism** — process-adapter JSONL (only stable cross-language path today, per-task overhead) **vs** in-process C-ABI / a future PyO3 binding. Gates the GIL/async (ADR-15) + perf work.
3. **Backward-compat posture** — sanction ADR-14 "managed-debt" shims for the migration window (conflicts with nirs4all/CLAUDE.md's absolute *no-shims* rule)?
4. **Sequencing** — god-class decomposition + gold-baseline capture **before** the bridge, or in parallel? (Externalization notes recommend extract-protocols-first.)
5. **v1 scope** — basic surface (baseline + linear/branch/merge/stacking/generators) only, or include heterogeneous multi-source repetitions (N0–N8/D0–D6) from day one?
6. **Numerics ownership** — run nirs4all operators as-is via the sklearn process adapter, or pursue the nirs4all-methods/libn4m swap concurrently?
7. **Tracked vs local docs** — several dag-ml core docs are gitignored/local-only (`.gitignore` 23–32). Commit them (so the migration plan is versioned), or keep local?
8. **Perf gate** — acceptable end-to-end overhead vs legacy (ADR-17 dual-run) before flipping the default backend?

---

*Recon provenance: `5f20f507…/tasks/w3tix7niq.output` (full structured synthesis), 2026-06-23.*
