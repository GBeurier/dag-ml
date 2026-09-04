---
orphan: true
---

# Developer documentation map

This is the versioned navigation map for contributors and coding agents. It
does not depend on private files being present in a fresh clone.

## Current contracts and verification

| Need | Source of truth |
|---|---|
| Product ownership and invariants | [Coordinator specification](COORDINATOR_SPEC.md) |
| Supported release surface and exclusions | [Support matrix](SUPPORTED.md), `CHANGELOG.md` at repository root |
| Architecture and ABI ownership | [Architecture](ARCHITECTURE.md), [C ABI](ABI.md) and the shipped headers |
| Accepted architecture decisions | [ADR index](adr/README.md) |
| Schemas and shared fixtures | [Contracts](contracts/README.md) |
| Native training and replay | [Training](TRAINING_CONTRACTS.md), [Replay](TRAINING_REPLAY_CONTRACTS.md) |
| Validation gates | `CONTRIBUTING.md`, `.github/workflows/ci.yml`, `scripts/`, crate tests |
| Runnable examples | `examples/README.md` at repository root |

Audit or acceptance claims must identify the release tag or exact source SHA,
commands actually run and any missing platform/runtime checks. A historical
"passed" statement is not evidence for a later release.

## Historical development evidence

The [original design inventory](design/README.md) points to preserved drafts
under `design/source/_archive/`. The [migration study](migration-nirs4all/README.md)
retains the June 2026 investigation and links to proposals. Its dated preparation
status and open decisions must be reconciled with current ADRs and code.
The native persistence report in that directory also contains later implementation
notes; use the dates and section status rather than treating the whole directory
as an outstanding backlog.

Keep the original evidence when archiving a document. Mark its date, former
location and successor; update active navigation links. Do not rewrite old review
results to make them appear to validate new code.

## Private records

Local specifications, reviews, audits and AI work records that have not been
published belong under `docs/_private/`, excluded from Git and the documentation
build. A local `README.md` should index `archive/` records and active work by
topic/date. This directory is optional and must never be force-added.

The former local-only `TOC.md`, `STATUS.md`, `ROADMAP.md`, `TEST_PLAN.md`, MVP
acceptance, final-release audit and host/WASM/multisource backlogs are historical
records, not required contributor inputs. Existing public ADRs, schemas and
design evidence remain versioned in their established directories.
