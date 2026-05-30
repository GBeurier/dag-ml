# Architecture Decision Records (ADRs)

Phase 0 of the nirs4all integration roadmap (`/home/delete/.claude/plans/use-4-agents-et-immutable-axolotl.md`) closes 18 load-bearing decisions before any broad workstream can start. Each decision lives here as a numbered, frozen ADR.

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

Format: each ADR is one page max, structured **Status / Context / Decision / Consequences / Blocks**. Changing a decision requires a new ADR that supersedes the old one (and explicitly says so).

The `dag-ml-data` sibling repo carries copies of ADRs 01, 02, 05, 06, 07 — the ones where the data layer is the primary enforcement site — under `dag-ml-data/docs/adr/`. The two ADR sets must stay byte-identical for the shared ADRs; CI validates this drift via `scripts/validate_contracts.py`.
