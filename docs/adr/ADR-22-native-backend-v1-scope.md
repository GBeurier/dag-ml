# ADR-22: Native-backend V1 scope and portability boundary

**Status**: accepted (2026-08-20)
**Supersedes**: none. It clarifies the non-normative V1 audit and roadmap;
ADR-17 remains in force.
**Blocks**: R1 contract freeze, capability ledger, archive V1, native-default cutover,
Studio sidecar, and every `DROP-*` work item.

## Context

The ecosystem has three partially overlapping execution descriptions: the Python
oracle and its legacy scheduler, the Rust DAG/data control plane, and the
portable `libn4m` numerical engine.  Treating any one of them as the whole
backend hides either host-controller dependencies or a Python fallback.  It
also makes the phrases "native backend" and "Rust-only" ambiguous.

V1 must preserve the public Python façade while making portability decisions
before data loading, training, writing or another expensive action.  Studio
must be distributable without a FastAPI/Uvicorn or CPython backend.  The
existing ADR-17 retention and rollback guarantees are still accepted and
cannot be bypassed by this decision.

## Decision

1. **Native backend** has explicit ownership. `dag-ml` owns CV/OOF
   orchestration, native aggregation (`avg`, `w_avg`, and `final`), metrics,
   selection/refit, lineage, and the generic prediction/score store.
   `dag-ml-data` owns typed data views and identity; `nirs4all-io` owns dataset
   assembly; and `nirs4all-core` composes and exposes those upstream contracts
   without reimplementing them.
2. Portable NIRS numerical computation remains in C++17 `libn4m` behind its
   versioned C ABI. `nirs4all-methods` owns numerical kernels/models, that ABI,
   N4MM, and the N4MOPT optimizer engine (ask/tell, samplers and pruners). A
   Rust Methods controller adapts it to `dag-ml`; `dag-ml` retains trials,
   folds, influence, selection and refit. Model binaries outside Methods remain
   host/plugin-owned. V1 does not rewrite `libn4m` in Rust or reimplement its
   optimizer in `dag-ml`.
3. The public Python API remains a compatibility façade. Python, sklearn, DL,
   SHAP or other non-portable controllers are allowed only as explicitly
   declared plugins. They are never an implicit orchestration backend and
   cannot be selected as a transparent fallback.
4. Every public API, model, operator and format has one preflight disposition
   in `nirs4all-ecosystem/docs/contracts/release/native-capability-ledger.v1.json`.
   Each entry is indexed by capability kind, stable identifier and
   language/runtime, and is one of `native`, `plugin`, `refused`, or
   `not-promised`. In the strict profile, an unknown entry, `not-promised`, a
   missing plugin, and `refused` are typed refusals before consuming scientific
   data, writing a result, or incurring meaningful compute. No disposition has
   an implicit executable default. The existing public surface matrix remains a
   separate release-accounting input; CAP-001 must add fixtures, cross-reader
   validation, and an ADR-02-compatible additive evolution or migration.
5. During the native-default release and the two subsequent ADR-17 retention
   releases, the rollback-capable product installer continues to carry
   `backend="legacy"` in the same installation. A separate strict profile may
   be tested during that window, but it cannot replace the rollback-capable
   product. Only after the retention window and `DROP-*` gates may the sole
   distributed product be fail-closed. Studio then launches a Rust sidecar and
   may invoke an explicitly installed external plugin only through the
   capability contract. React/Electron and the Python façade are not themselves
   evidence of a Python backend.
6. `dag-ml`'s generic prediction/score store is not the product
   workspace/session/archive. The product aggregate (`nirs4all-core`, with the
   precise writer named by `SAVE-001`) owns that product boundary and must
   preserve the public `session()` adapter contract. `SAVE-001` names the single
   archive writer, versions, dual readers, migration direction, and rollback
   before any default flip; this ADR changes no existing wire.
7. The Python 0.12.0 line remains an archived behavioral oracle, including its
   fixtures and readers needed by transition tools, until `LOCK-DROP` is
   proven. It is not a runtime fallback in a fail-closed product.
8. V1 excludes Cluster production, Tauri migration, a Rust rewrite of Methods,
   new algorithm families, multimodal productisation, full shared-chart
   adoption, conformal extensions beyond the required split-regression
   integration, and finetuning extensions unless one closes a frozen
   compatibility gate.
9. ADR-17 is not superseded. A native-default release is followed by two full
   releases in which legacy remains available with rollback without
   reinstallation and bundle readability. `DROP-*` remains blocked until that
   window is complete. A later ADR may supersede ADR-17 only by stating the
   replacement rollback, bundle-readability and release-count guarantees
   explicitly.

## Consequences

- The authoritative V1 capability ledger and release lock live with the
  ecosystem release contracts; component-local capability files remain inputs,
  not competing authorities.
- A contract change crossing data, methods, archive, public API or Studio
  boundaries requires its owning repository, fixtures, compatibility direction,
  targeted tests and an independent review before consumers advance.
- `nirs4all-core` may expose upstream runtimes and bindings, but parsers,
  numerical kernels, dataset catalog logic and DAG scheduling stay in their
  owning projects.
- `DROP-*` cannot start merely because native-default builds exist. It still
  requires the ADR-17 retention window, `LOCK-DROP`, and the release gates.

## Blocks

`SAVE-001`, `MTH-001`, `DATA-001`, `STU-001`, `CAP-001`, `PAR-002`, the
required split-regression conformal integration, and all R2/R3 cutover work
consume this decision. Any exception must be recorded as a new ADR and
reflected in the ecosystem capability ledger and aggregation lock.
