# ADR-19: Multi-source unit vocabulary and derived-combo decision

**Status**: accepted (2026-06-13)
**Blocks**: heterogeneous multi-source repetitions roadmap, phases D1–D8. Land BEFORE any D1–D8 schema change; gates the deferred public prediction-level decision.

## Context

`docs/HETEROGENEOUS_MULTISOURCE_REPETITIONS_ROADMAP.md` extends the coordinator
to pipelines where one physical sample carries several observations per source
with asymmetric cardinalities across sources (e.g. `A=2/B=3/C=2`). Before any
schema work (D1–D8), the unit vocabulary and the migration posture must be
frozen so downstream phases share one set of terms and none of them silently
promotes a breaking public prediction level. The core stays a coordinator: no
feature-buffer materialization, no fitted-model internals; identities,
relations, reducers, OOF/fold safety, selection, lineage and replay only.

## Decision

### Frozen vocabulary

These names are canonical for D1–D8 and follow-up work. Renames require an ADR-superseder.

| Term | Meaning |
|---|---|
| `physical_sample` | Logical sample requested by the user; the leakage, target and default output unit. Owns all its source observations and repetitions. Aligns with `SampleId`. |
| `source_sample` | Per-source view of one physical sample (its observations within a single source). Intermediate domain for source-aware models/late fusion; not a public prediction level in the mainline. |
| `observation` | One physical row/acquisition in one source. Aligns with `ObservationId`; many observations may share one `physical_sample`. |
| `combo` | Derived observation/unit built by combining one observation (or aligned tuple) per source for the same physical sample. Stored relation-backed with `component_observation_ids` and `origin_sample_id`. |
| `EntityUnitLevel` | Internal enum naming the unit domain of a relation row, port, reducer or view: `physical_sample`, `source_sample`, `observation`, `combo`. Additive Rust type and schema vocabulary; distinct from the public `PredictionLevel`. |
| `PredictionUnitId` | Stable identity of a non-sample prediction row (already used for target/group caches). Carries derived/relation unit ids for combos without adding a public combo prediction level. |
| `ReductionPlan` | Contract for moving values across unit levels/axes (`role`, `axis`, `input_unit_level`, `output_unit_level`, `method`, `weight_source`). A `combo`/`observation -> physical_sample` reduction is mandatory for any row-multiplying representation to yield sample output. |
| `RepresentationPlan` | Contract for a host-built feature representation (aggregate, cartesian, Monte-Carlo cartesian, stack fixed, stack padded/masked) plus the relation delta, replay manifest, cardinality contract and missing-source policy it emits. Core plans contracts and host handles only. |
| `FitInfluencePolicy` | Contract for how per-sample influence is enforced at fit time (`auto`, `uniform_rows`, `equal_sample_influence`, `resample_equalized`, `backend_loss_weight`, `scorer_only`, `strict_weight_support`). Distinct from `AggregationWeights`: influence is a fit contract, not an aggregation weight. |

### Mainline: combo is a derived observation, not a public prediction level

In the mainline implementation (D1–D8) a `combo` is a relation-backed derived
observation/unit. The public prediction-cache levels stay `observation`,
`sample`, `target`, `group`; `policy::PredictionLevel` gains no `combo` or
`source_sample` variant. Cartesian/Monte-Carlo training, combo-to-sample
reducers, replay and audit operate through the relation table
(`component_observation_ids`, `origin_sample_id`) and the existing sample-level
OOF/aggregation paths. Final public output remains sample-level.

### Deferred gate: first-class public combo / source_sample levels

Promoting `combo` and/or `source_sample` to first-class public `PredictionLevel`
/ `PredictionUnitId` values is deferred until a separate ADR explicitly
approves the public-contract change. It is needed only when a downstream public
prediction cache, selector or meta-model consumes combo/source rows directly.
Approval must carry: cache/schema migration notes; an ABI/binding impact review
across prediction-cache metadata schemas, `PredictionBlock`, JSON fixtures,
CLI/Wasm validation, Python bindings and the C ABI snapshot; and compliance with
ADR-02 and ADR-14 (versioning, dual-read or migration edge, CHANGELOG, negative
fixtures). Until then the relation-backed `meta_row_domain=combo` path covers
combo meta-models without a public combo cache.

### ADR-02 migration checklist (this feature)

Every D1–D7 phase that changes a public type applies ADR-02 in that same phase
(D8 audits coherence; it is not a catch-up phase):

1. **Optional fields first** — new relation/port/edge/reducer/selection/cache
   fields land as `Option<T>` with a documented default; v1 readers ignore
   unknown fields per `additionalProperties`.
2. **Defaults / dual-read** — defaults reproduce current sample-level behavior;
   relation-aware validation is stricter only when relation metadata is present;
   old fixtures still validate or are dual-read migrated.
3. **Fixture update** — the schema JSON, local fixture and conformance pack move
   with the Rust type; cross-repo parity is re-checked with `DAG_ML_DATA_REPO`
   when shared contracts change.
4. **CHANGELOG entry** — every wire-shape change is recorded under ADR-02 (and
   ADR-14 for any deprecation).
5. **C ABI decision** — each phase explicitly records "ABI touched (bump
   `abi_snapshot.v1.json` + `dag_ml.h` + tests)" or "ABI not touched";
   relation/prediction-unit tables cross the ABI only if the deferred
   first-class-combo gate is later approved.

## Consequences

- Downstream phases use derived-observation combos by default; the first-class
  public combo level is isolated as one deferred public-contract decision.
- The vocabulary is surfaced in `docs/COORDINATOR_SPEC.md`,
  `docs/ARCHITECTURE.md` and `docs/contracts/README.md` so schema and Rust work
  share one glossary.
- The roadmap's public-contract matrix plus this per-phase ADR-02 checklist are
  the audit basis for D8.

## Risk

- A premature first-class `combo` level would force a breaking
  prediction-cache/ABI migration. Mitigation: the relation-backed mainline plus
  the explicit deferred gate keep that change opt-in and reviewed.
