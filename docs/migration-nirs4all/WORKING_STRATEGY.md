# Working strategy — migrate the core while keeping prod alive

> Answers ask #3: *fork? branch? new repo?* — and how to keep maintaining prod nirs4all
> locally **and** remotely throughout a multi-month chantier.
>
> **PROPOSAL — pending open decisions #1/#2/#7 in [`README.md`](README.md).**

## Constraints this must respect

- `nirs4all` `main` is **production** (tags 0.10.x; webapp depends on the frozen 0.9.x surface). It must keep shipping hotfixes/features the whole time.
- The team already works **feature-branch-on-one-repo** (`GBeurier/nirs4all`): `cache_refactoring`, `feat/heterogeneous-repetitions`, `parquets_refactoring`, … This migration should ride that workflow, not fight it.
- `dag-ml` is a **separate, contract-frozen submodule** (0.2.x RC, currently 0.2.5) co-evolving with `dag-ml-data` under the ADR-10 release train. It is wheel-buildable/importable from the built PyO3 wheel; PyPI publishing is a configured release target.
- nirs4all/CLAUDE.md mandates **phased execution** (≤5 files/phase, dead-code cleanup committed separately first) and **no long-lived backward-compat shims** (ADR-14 wants a temporary exception — open decision #3).

## Recommendation — *not* a fork, *not* a new repo: **selector-on-main + integration branch + worktree**

The load-bearing insight is **ADR-17**: a runtime **backend selector** `engine ∈ {legacy, dag-ml, dual}` that **defaults to `legacy`**. With that flag, migration code can be merged into `main` *incrementally* without changing a single production code path — prod simply never flips the flag. This collapses the classic "giant long-lived branch that rots" problem.

So the model is three layers:

### 1. Destination = `dag-ml` directly (recommended), consumed as an optional dependency
- Add a `nirs4all[dagml]` extra pinning a compatible `dag-ml` (and `dag-ml-data`) version. The import is already guarded-style elsewhere; the engine import lives behind the selector and is **never imported when `engine=legacy`**, so base install is unaffected.
- **Reject** the new-repo / fork path and the "pipeforge" generic-lib detour for the *core swap*: they fragment the community, double the maintenance, and the recon shows dag-ml — not pipeforge/v2 — is the authoritative, already-built target. (Keep pipeforge/v2 docs as parity input only.)

### 2. Integration lives on `main` behind the selector, in flag-gated increments
- New code path: `nirs4all/pipeline/engine/` (or similar) holding the dag-ml bridge (DSL serialization → `dag_ml` importer, host process-adapter, result re-hydration). Selected by `engine=` kwarg / `N4A_ENGINE` env / config — **default `legacy`**.
- Each PR is small (≤5 files), parity-gated (see [`PARITY_AND_PERF_HARNESS.md`](PARITY_AND_PERF_HARNESS.md)), and merges to `main` because it is inert at `engine=legacy`. Prod stays green; rollback = "don't set the flag" (zero-cost, per ADR-17).
- The **god-class decomposition** (orchestrator/merge/branch/base_model/SpectroDataset) is *pure refactor* that benefits prod too — it lands on `main` as normal, behind the project's phased rule, **before** the bridge needs to cross those seams. This is the "extract-protocols-first" order the externalization notes recommend.

### 3. A long-lived branch + worktree **only** for the churny, not-yet-flag-isolatable spikes
- For exploratory work that can't yet be made inert (e.g. the first end-to-end bridge spike, serialization-format experiments), use **one** integration branch `core/dagml`, rebased on `main` frequently, and collapsed into flag-gated PRs as soon as a slice is parity-passing. Don't let it accumulate.

### Local setup — maintain prod and migrate at the same time, zero context-switch

Use a **git worktree** so prod `main` and the migration branch are *two directories at once*:

```bash
# from the existing prod checkout
cd /home/delete/nirs4all/nirs4all
git worktree add ../nirs4all-core core/dagml      # second working dir on the migration branch
# prod hotfixes:   work in   nirs4all/        (main)
# migration work:  work in   nirs4all-core/   (core/dagml)

# editable, co-evolving dag-ml in the migration venv only:
cd ../nirs4all-core && python -m venv .venv && . .venv/bin/activate
pip install -e '.[dagml,all]'
maturin develop -m ../dag-ml/crates/dag-ml-py/Cargo.toml --release   # local editable engine
```

Prod venv (`nirs4all/.venv`) never gets `dag-ml` → it cannot accidentally change behaviour.

### Remote & release

- **Prod releases** continue from `main` exactly as today (ADR-10 release train; `engine` defaults to `legacy`, so 0.10.x/0.11.x ship with dag-ml dormant).
- **Migration CI**: push `core/dagml`; CI runs the parity suite in `dual` mode + perf benchmarks (see harness doc). PRs to `main` run the fast parity-compile gate + legacy baseline.
- **Cutover** is then just a normal PR that flips the default once the parity + perf gates are green for the targeted scope — and is reversible by flipping back.
- **Cross-repo pinning**: extend the ecosystem meta-repo with a tested `(nirs4all branch/rev × dag-ml rev × dag-ml-data rev)` compatibility triple per ADR-10, so the three stay in lockstep.

## Why not the alternatives

| Option | Verdict |
|---|---|
| **Fork `nirs4all`** | ✗ Splits issues/PRs/community; merge-back is a nightmare; no upside over a flag on `main`. |
| **New repo `nirs4all-next` / `pipeforge`** | ✗ Only justified if destination = the generic-core detour (open decision #1), which the recon argues against. Doubles maintenance. |
| **One giant long-lived branch** | ✗ Rots against a `main` that ships weekly; the selector makes it unnecessary. |
| **Selector-on-main + worktree (recommended)** | ✓ Prod inert by default, incremental & reviewable, zero-cost rollback, matches existing workflow, decomposition benefits prod immediately. |

## First concrete steps (once decisions #1/#2 are set)
1. Land the **backend-selector skeleton** on `main` (`engine=legacy|dag-ml|dual`, default legacy; `dag-ml` path raises `NotImplemented` for now). Inert, tiny, unblocks everything.
2. Stand up the **parity gold-baseline + dual-run hook** (harness doc) — needed before any bridge code is trustworthy.
3. Begin **god-class decomposition** along the dag-ml seam (phased, ≤5 files, cleanup-first), starting with the orchestrator/executor OOF + refit boundary.
4. Build the **DSL serialization frontend** (live pipeline objects → portable JSON descriptors the `dag_ml` importer accepts) — the #1 dag-ml-side gap.
