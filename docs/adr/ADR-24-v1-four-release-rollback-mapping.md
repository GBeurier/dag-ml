# ADR-24: V1 four-release rollback mapping

**Status**: accepted (2026-08-21)
**Supersedes**: ADR-17 decision items 3 and 4, and ADR-22 decision items 5 and 9,
for the R1–R4 V1 release train only.
**Blocks**: `ARCH-002`, `LOCK-DROP`, every `DROP-*` item, and the R3/R4
installer claims.

## Context

ADR-17 requires the legacy backend to remain available for two full releases
*after* the native-default flip, with rollback in the same installation and no
bundle migration.  The V1 roadmap contains exactly four releases: R1
consolidation, R2 native default, R3 strict product candidate, and R4 stable.
Treating R3 as physical removal of every legacy byte would violate ADR-17;
counting builds or the flip release as a retention release would weaken its
explicit guarantee.

The ambiguity must be resolved before claiming either R3 fail-closed behavior
or R4 stability.

## Decision

1. **Release count.** R2 (`1.0.0-rc.1`) is the native-default flip.  R3
   (`1.0.0-rc.2`) and R4 (`1.0.0`) are the two complete post-flip retention
   releases.  Release candidates, rebuilds, CI artifacts, and nightly builds
   never count as releases.
2. **Strict product path.** In R3 and R4, the default native path is
   fail-closed: unsupported or plugin-only capabilities are refused before
   scientific-data access, result writing, or meaningful computation.  It
   never transparently reruns the Python legacy runner.  Studio ships and
   launches only the Rust sidecar; it contains no CPython/FastAPI fallback.
3. **Rollback profile.** The Python compatibility distribution carries the
   explicitly selected `engine="legacy"` rollback profile through R4.  It is
   not a default, not a transparent fallback, and is labelled as rollback-only
   in APIs, installers, capability evidence, and release notes.  Thus a user
   can roll back execution without reinstalling or migrating archives while
   the strict product and Studio paths remain Python-free.
4. **Bundle readability.** Archives created by either profile remain readable
   for PREDICT during R2–R4.  Conversion is explicit, verified, and never
   required merely to select the rollback profile.
5. **Removal.** Physical removal of the legacy compatibility profile is a
   post-V1 release.  It may start only after R4 has shipped, `LOCK-DROP` and
   every `DROP-*` gate are green, and an explicit removal release records the
   version in the changelog.  R3/R4 must not claim that the rollback profile
   has been removed.

## Consequences

- The roadmap phrase “fail-closed” means native-default and Studio behavior;
  it does not authorize an implicit fallback or an early deletion of the
  explicit Python rollback profile.
- R3/R4 installers and web pages must distinguish the Rust-only Studio product
  from the rollback-capable Python compatibility distribution.
- The capability ledger retains `rollback_profile.backend=legacy` and
  `retention_releases=2`; its evidence must name this ADR.
- `ARCH-002` is resolved, but `LOCK-DROP` remains pending until after R4.

## Blocks

R3/R4 release notes, installer manifests, Studio packaging, public API engine
selection, capability-ledger validation, and all legacy removal work consume
this decision.
