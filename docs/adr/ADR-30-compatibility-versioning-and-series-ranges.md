# ADR-30: Compatibility versioning and tested-series ranges

**Status**: accepted (2026-11-18, maintainer rule)
**Supersedes**: ADR-29 decisions 1 and 3 (1.x range forms, patch/minor meaning) and its "no patch floor" consequence.
**Blocks**: all releases.
**Scope**: every `dag-ml` release and every dependency of a `dag-ml` package on a package from another repository.

## Context

ADR-29 replaced exact pins with minor-level ranges, but its 1.x forms
(`"1.3"`, `>=1.3,<2`, `^1.3.0`) accept every later minor, and it let a 1.x minor
add ABI/API surfaces. The maintainer's ecosystem rule of 18 November 2026 ties
the version number to compatibility instead.

## Decision

1. Versions are `X.Y.Z`. `Z` (patch) carries fixes and purely additive,
   backward-compatible features; existing outputs, wire formats, schemas and
   ABI symbols stay unchanged. `Y` (minor) carries any compatibility break:
   changed/removed API signature, C ABI break or ABI version change,
   wire/byte/schema/fixture change (including `docs/contracts/` and conformance
   packs), renamed ids, or changed numerical results of existing operations.
   `X` is the product version. The bumping package classifies its change.
2. Cross-repository dependencies accept one tested series:
   `>=X.Y[.Z],<X.(Y+1)`. A patch floor is allowed when a fix or feature from
   that patch is required. Moving to a new `Y` is an explicit, tested consumer
   decision. For 1.x packages never use npm `^1.x` or Cargo `"1.x"`: write
   `~1.4.1`, `">=1.4.1, <1.5"`, `>=1.4.1,<1.5`. For 0.x, `"0.4"`, `^0.4.0` and
   `>=0.4,<0.5` already stop at the next minor. R declares a lower bound and a
   documented series.
3. ADR-29 decisions 2, 4 and 5 are unchanged: exact versions stay in lockfiles,
   intra-repository crates/bindings stay in lockstep, frozen oracles keep their
   exact pins, and the runtime ABI check (not the crate version) accepts a
   native `n4m` library.

## Consequences

- Existing ranges that predate this rule (for example `nirs4all-methods>=1.3,<2`
  in the `multimodal` extra) are aligned at the next normal release.
- `scripts/validate_release_metadata.py` still requires a `MAJOR.MINOR` `n4m`
  range; `n4m` is 0.x, so that form stops at the next minor.
- The full ecosystem policy is `docs/contracts/release/VERSION_CONSTRAINTS.md`
  in `nirs4all-ecosystem`.
