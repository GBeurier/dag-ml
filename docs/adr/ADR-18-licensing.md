# ADR-18: Licensing (CeCILL-2.1 vs MIT)

**Status**: proposed (2026-05-29) — **requires maintainer decision**
**Blocks**: workstream C, workstream D, all releases

## Context

`nirs4all` is licensed CeCILL-2.1. `dag-ml` and `dag-ml-data` are MIT. When nirs4all takes a hard dependency on the MIT crates/wheels, the redistribution obligations of the combined work must be analyzed. This is load-bearing for any release that ships nirs4all together with the dag-ml backend.

This ADR is **proposed**, not accepted: it documents the options and requires the maintainer (Guilhem) to choose. Until a decision lands, releases proceed under path (c) by default.

## Decision (options — maintainer must select one)

**(a) Keep current licenses + thin CeCILL-compatible wrapper.** nirs4all stays CeCILL-2.1 and depends on MIT dag-ml as an external runtime dependency. MIT is permissive and CeCILL-2.1 is GPL-compatible, so MIT-into-CeCILL is generally fine, but the combined-work obligations warrant a one-page legal read before this is declared safe.

**(b) Dual-license dag-ml / dag-ml-data as `MIT OR CeCILL-2.1`.** Lets nirs4all incorporate dag-ml source under matching terms if source vendoring is ever needed. Costs: every contributor must agree to dual-licensing; the `LICENSE` files and crate metadata change.

**(c) Explicit non-incorporation (recommended default).** nirs4all consumes dag-ml strictly as a PyPI / crates.io **runtime dependency** — no source vendoring, no static linking of source into the nirs4all distribution. The license boundary stays clean: each project ships under its own license and users install both. Lowest friction; recommended unless source vendoring becomes necessary.

## Consequences

- Workstream G task 4 produces a one-page redistribution analysis once a path is chosen; this ADR is then updated to **accepted** with the selected path and the analysis linked.
- Until then, the release train (ADR-10) operates under path (c): dag-ml and dag-ml-data are published as independent dependencies, never vendored into nirs4all's source tree.
- Crate metadata (`license = "MIT"`) and nirs4all's `license = "CECILL-2.1"` remain unchanged under path (c).

## Risk

- Choosing (a) or (b) later, after path (c) has shipped, is non-breaking for users (they already install both packages) but requires the legal read and, for (b), contributor re-licensing consent. Deferring the decision is therefore low-risk as long as path (c) is the operating default.
