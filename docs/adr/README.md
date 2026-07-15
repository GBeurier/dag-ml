# Architecture Decision Records (ADRs)

Phase 0 of the nirs4all integration roadmap closes the load-bearing decisions before any broad workstream can start. Each decision lives here as a numbered, frozen ADR; ADR-01 through ADR-18 form the frozen Phase-0 baseline, and ADR-19 onward extends it for later feature roadmaps.

| # | Title | Status | Blocks |
|---|---|---|---|
| 01 | [Compatibility ledger semantics](ADR-01-compatibility-ledger.md) | accepted | B, E, F |
| 02 | [Schema evolution SLA](ADR-02-schema-evolution-sla.md) | accepted | B |
| 03 | [Separation-branch semantics](ADR-03-separation-branch-semantics.md) | accepted | E |
| 04 | [Tag / exclude mask materialization](ADR-04-tag-exclude-masks.md) | accepted | E, G |
| 05 | [Repetition-aware CV invariant](ADR-05-repetition-cv-invariant.md) | accepted | B, E, F |
| 06 | [Signal-type ownership](ADR-06-signal-type-ownership.md) | accepted | B, E |
| 07 | [Aggregation reducers in contract](ADR-07-aggregation-reducers.md) | accepted | B, E |
| 08 | [Session / workspace persistence](ADR-08-session-persistence.md) | accepted | E |
| 09 | [Docs stack](ADR-09-docs-stack.md) | accepted | C |
| 10 | [Cross-repo release train](ADR-10-release-train.md) | accepted | C, D |
| 11 | [Unified error taxonomy](ADR-11-error-taxonomy.md) | accepted | A, D, E, G |
| 12 | [Observability hooks](ADR-12-observability-hooks.md) | accepted | A, G |
| 13 | [Process-adapter security boundary](ADR-13-process-adapter-security.md) | accepted | A, G |
| 14 | [Public-API deprecation policy](ADR-14-deprecation-policy.md) | accepted | C, E |
| 15 | [Python GIL / async](ADR-15-gil-async.md) | accepted | D, E |
| 16 | [Artifact serialization security](ADR-16-artifact-security.md) | accepted | A, E |
| 17 | [Feature flag / cutover / rollback](ADR-17-cutover-rollback.md) | accepted | E, F |
| 18 | [Licensing](ADR-18-licensing.md) | proposed | C, D, all releases |
| 19 | [Multi-source unit vocabulary & derived-combo decision](ADR-19-multisource-unit-vocabulary.md) | accepted | multi-source roadmap |
| 20 | [Native conformal calibration ownership and identity boundary](ADR-20-conformal-calibration-ownership.md) | accepted | conformal W0-W4 |
| 21 | [Public training replay ownership and port-explicit wire evolution](ADR-21-forward-replay-ownership.md) | accepted | replay D4-D9 |

ADR-19 onward extends the registry for feature roadmaps that build on this
Phase-0 baseline; ADR-19 freezes the unit vocabulary and migration posture for
the heterogeneous multi-source repetitions roadmap
(`docs/HETEROGENEOUS_MULTISOURCE_REPETITIONS_ROADMAP.md`).

Format: each ADR is one page max, structured **Status / Context / Decision / Consequences / Blocks**. Changing a decision requires a new ADR that supersedes the old one (and explicitly says so).

The `dag-ml-data` sibling repo carries copies of ADRs 01, 02, 05, 06, 07 — the ones where the data layer is the primary enforcement site — under `dag-ml-data/docs/adr/`. The two ADR sets must stay byte-identical for the shared ADRs; CI validates this drift via `scripts/validate_contracts.py`.

```{toctree}
:maxdepth: 1
:hidden:

ADR-01-compatibility-ledger
ADR-02-schema-evolution-sla
ADR-03-separation-branch-semantics
ADR-04-tag-exclude-masks
ADR-05-repetition-cv-invariant
ADR-06-signal-type-ownership
ADR-07-aggregation-reducers
ADR-08-session-persistence
ADR-09-docs-stack
ADR-10-release-train
ADR-11-error-taxonomy
ADR-12-observability-hooks
ADR-13-process-adapter-security
ADR-14-deprecation-policy
ADR-15-gil-async
ADR-16-artifact-security
ADR-17-cutover-rollback
ADR-18-licensing
ADR-19-multisource-unit-vocabulary
ADR-20-conformal-calibration-ownership
ADR-21-forward-replay-ownership
```
