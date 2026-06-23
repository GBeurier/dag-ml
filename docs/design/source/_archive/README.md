# `design/source/_archive/` — superseded design-source drafts

These are the **genetic design drafts** of dag-ml. They are **superseded as normative
contracts** (the live source of truth is [`../../COORDINATOR_SPEC.md`](../../COORDINATOR_SPEC.md)
+ [`../../TOC.md`](../../TOC.md) + the ADRs), but they are kept because they carry
**irreplaceable migration provenance** — they were the bridge between the nirs4all engine
and the dag-ml design and contain the most detailed code-level migration orders that exist.

Do **not** plan field-level work against these files; reconcile names/signatures against the
current `docs/` contracts and `crates/*/include/*.h` headers first. Read them for *intent and
sequencing*, not for the current API.

| File | What it still gives you | Superseded by |
|---|---|---|
| `dag_ml_specification_v1.md` | Full engine design; **§19 = a 10-step nirs4all migration plan** + DSL→NodeKind lowering table | The shipped Rust 0.2.0 core + `COORDINATOR_SPEC.md` |
| `dag_ml_externalization_from_code.md` | Reverse-engineering of the *live* nirs4all engine: **exact files to extract + a concrete 9-step extraction order** | `COORDINATOR_SPEC.md`, `CAPABILITY_MATRIX.md` (post data-split) |
| `dag_ml_use_cases.md` | 11/12 use-cases are real nirs4all pipelines; **Annexe-A DSL-keyword→NodeKind table** = de-facto acceptance spec | `MVP_ACCEPTANCE.md`, `design/DSL_NIRS4ALL_PARITY.md` |
| `dag_ml_synthese.md` | Canonical entry rationale (objective / decisions / roadmap) | `COORDINATOR_SPEC.md`, `RATIONALE.md` |
| `dag_ml_polyglot_core_design.md` | Rationale for operators-host-side + frozen vtables / RNG tiers (Rust-vs-C++ left *open* at the time) | `ABI.md` + shipped C-ABI headers |

The migration war-room that cross-links these for the nirs4all-core effort is at
[`../../migration-nirs4all/`](../../migration-nirs4all/).
