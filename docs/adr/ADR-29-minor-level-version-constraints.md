# ADR-29: Minor-level cross-repository version constraints

**Status**: accepted (2026-11-18)
**Amended by**: [ADR-30](ADR-30-compatibility-versioning-and-series-ranges.md) (2026-11-18, maintainer rule) — 1.x ranges stop at the next minor (`>=X.Y[.Z],<X.(Y+1)`), a patch floor is allowed when required, and any compatibility break, including an additive ABI version change, needs a minor.
**Supersedes**: ADR-10 steps 2 and 4 (consumer pin bumps after every release) and its "version pinning becomes a hard CI constraint" consequence.
**Blocks**: all releases.
**Scope**: every dependency of a `dag-ml` package on a package from another repository, and every consumer's dependency on `dag-ml`.

## Context

Exact cross-repository pins (`n4m = "=0.4.2"`, `dag-ml = "=0.4.0"`, `==` in
Python) forced every consumer to release after each patch of a lower layer, only
to move a pin, and prevented candidate-cohort builds from patching an unreleased
crate whose version did not match an exact requirement.

## Decision

1. Cross-repository constraints are expressed at **minor** granularity, never
   more precisely: Rust `"0.4"` (caret, `>=0.4.0, <0.5.0`; `"1.3"` for 1.x),
   Python `>=0.4,<0.5` (`>=1.3,<2` for 1.x), npm `^0.4.0` (`^1.3.0`), R
   `pkg (>= 0.4.0), pkg (< 0.5.0)`. Here: `n4m = "0.4"` in `dag-ml-core` and
   `dag-ml-cli`, `nirs4all-methods>=1.3,<2` in the `dag-ml` Python
   `multimodal` extra.
2. Exact, reproducible versions live in lockfiles (`Cargo.lock`,
   `crates/dag-ml-py/Cargo.lock`, uv/pip locks, `package-lock.json`) and in the
   release order of ADR-10, which is otherwise unchanged.
3. A **patch** release of `dag-ml` is backward compatible and changes no wire
   format, byte contract, persisted schema (Archive/Package, replay, results),
   `.n4a` payload semantics, C ABI or shared contract/fixture under
   `docs/contracts/`; any such change needs at least a **minor** release (for
   0.x, every minor is breaking). Consumers therefore raise their range only for
   a new minor.
4. Crates and bindings of this repository stay in version lockstep. The isolated
   `tools/*-core-oracle` crates keep exact `=0.4.3`/`=0.4.6` Core pins: they are
   frozen scientific oracles, never published, not consumer constraints.
5. The native `n4m` library is accepted by its runtime ABI check, never by the
   crate version; the range only selects which binding is compiled.

## Consequences

- `scripts/validate_release_metadata.py` requires the `n4m` requirement to be a
  `MAJOR.MINOR` range and its locked version to fall inside it.
- A consumer needing a fix from a specific patch gets it through its lockfile
  and the release order, not a patch floor.
- No automated guard yet fails when a contract file changes without a minor
  bump; contract drift remains caught by `scripts/validate_contracts.py`, the
  ABI snapshot and the conformance packs.
- The full ecosystem policy is `docs/contracts/release/VERSION_CONSTRAINTS.md`
  in `nirs4all-ecosystem`.
