# Parity (non-regression) + performance harness — automated

> Answers ask #4: prove nirs4all-with-dag-ml == nirs4all-without-dag-ml (parity), measure the
> cost of each (performance), and automate both so the chantier runs under a green gate.
>
> **PROPOSAL.** Builds directly on the harness that already exists at
> `nirs4all:tests/integration/parity/` — we extend it, we don't rebuild it.

## What already exists (don't reinvent)

`nirs4all/tests/integration/parity/` is a **working scaffold explicitly built as the dag-ml
contract**, but the dag-ml side is not wired and the oracle isn't captured yet:

- **`_registry.py`** — frozen `PipelineCase` dataclass: `name, keywords, capabilities, dataset_key, pipeline_factory (→ fresh pipeline list), dataset_kwargs, task, expected_min_predictions, metric_tolerances, tags, skip_reason/skip_kind`. Import-time validation that `keywords ⊆ CANONICAL_KEYWORDS` and `capabilities ⊆ COMMON_CAPABILITIES`. ~35 cases across `cases_*.py` (baseline incl. **`baseline_vertical_slice`** the gate-zero case with the only tight tolerances `rmse/r2 = 1e-6`; branches_merges; 5 multi_source; aggregation_reps; augmentation; generators w/ expected variant counts; tags_exclude incl. leakage-aware; refit_predict public-API round-trips).
- **`test_parity_compiles.py`** — fast CI gate: every canonical DSL keyword has ≥1 case (allowlist documented).
- **`test_parity_smoke.py`** — runs `nirs4all.run/predict/explain/retrain/session` + `.n4a` export on the **legacy backend only**, asserting only `num_predictions >= expected_min_predictions`.

**The three missing pieces** (all called out by the recon): (1) no captured **gold baseline**, (2) `metric_tolerances` recorded but **unenforced**, (3) **no dag-ml backend hook** / dual-run.

## The plan — five layers, each independently valuable

### Layer 0 · Determinism contract (prerequisite)
Pin seeds in every `pipeline_factory` + `dataset_kwargs`; assert dag-ml's "sequential vs parallel byte-identical" guarantee also holds legacy-side for the captured cases. Without this, parity diffs are noise.

### Layer 1 · Capture the gold baseline (legacy = oracle of record)
Per ADR-01, the oracle is the **legacy backend's observed behaviour**. Add a capture mode that records, per case, a canonical **observation record**:
- prediction arrays → store summary stats + a quantized hash (full arrays only for `baseline_vertical_slice`),
- variant count (must equal `expected` for generator cases),
- **fold partition shape keyed by sample-id** (identity-keyed, not row index — dag-ml's invariant),
- best metric + the **score-key set** (ADR-01),
- `.n4a` round-trip: predictions reproducible after export→load→predict.

Store under `tests/integration/parity/baselines/<case>.json`, keyed by a content hash of `(serialized pipeline descriptor, dataset id, seed)` so a baseline auto-invalidates when the case changes. Capture command:
```bash
python -m nirs4all.testing.parity_capture            # or: pytest tests/integration/parity --parity-capture
```

### Layer 2 · Enforce tolerances **now** (legacy-vs-gold regression gate)
Flip `metric_tolerances` from recorded→enforced: re-run each case on legacy, diff vs its gold baseline within tolerance. **This pays off before dag-ml exists** — it catches accidental production regressions in ordinary nirs4all PRs. `baseline_vertical_slice` enforces the tight `1e-6`.

### Layer 3 · Dual-run parity (legacy vs dag-ml) — the cutover gate
Wire the **ADR-17 backend selector** into `test_parity_smoke.py`:
```python
@pytest.mark.parametrize("engine", ["legacy", "dag-ml"])   # "dual" diffs in-process
def test_case_parity(case, engine): ...
```
For each case run through dag-ml, diff vs the **gold baseline** within the **ADR-01 per-model-class tolerance table** (not a single global epsilon). Comparisons:
- predictions within per-model tolerance; **fold partitions exact** (identity-keyed); **variant counts exact**; selection decision exact; refused cases (UC11-style) must refuse identically.
- Gate which cases dag-ml is *expected* to pass via the existing `capabilities`/`tags` + the coverage in `dag-ml:docs/design/DSL_NIRS4ALL_PARITY.md`: uncovered constructs are `xfail(strict=True)` and flip to must-pass as the bridge lands. No silent skips — every skip prints its reason and the 4 known legacy-bug cases stay flagged.

### Layer 4 · Performance harness (the named cutover risk)
dag-ml ships only two ignored ~1.5 s sanity probes (`dag-ml:docs/PERFORMANCE.md`) — **promote them to repeatable benchmarks** and add end-to-end campaign benchmarks nirs4all actually cares about. New `tests/integration/parity/bench/` (pytest-benchmark or asv):
- per-case **wall-time + peak memory** for `legacy` / `dag-ml` / `dual`,
- **isolate the per-task JSONL process-adapter overhead** (the biggest unknown) — measure fixed cost/task and amortization across folds×variants,
- two sizes: tiny fixtures (CI) + large synthetic spectra × many folds/variants (nightly),
- store baselines; fail on regression beyond a budget; report the **dag-ml/legacy overhead ratio** that feeds the open-decision-#8 cutover threshold.

### Layer 5 · Automation / CI

| Trigger | Job | Budget |
|---|---|---|
| every commit (any branch) | `test_parity_compiles` + Layer-2 legacy-vs-gold (quick subset) | < 1 min |
| PR → `main` | full Layer-2 legacy regression gate | minutes |
| PR → `core/dagml` + nightly | full **Layer-3 dual-run** parity | minutes–tens |
| nightly | **Layer-4** perf bench + regression check | longer |
| ecosystem repo | ADR-10 **compatibility-triple** matrix `(nirs4all rev × dag-ml rev × dag-ml-data rev)` | nightly |

Add pytest markers `parity` + `bench` (alongside existing `slow`/`stress`) in `nirs4all:pyproject.toml`, and a one-liner local entry point (`nirs4all parity [--capture|--dual|--bench]` or `make parity`) so the loop is `edit → make parity → green`.

## Sequencing
1. **Layers 0–2 first** (no dag-ml needed): determinism + capture + enforce → immediate regression safety for prod.
2. Layer 3 lands with the **backend-selector skeleton** (see [`WORKING_STRATEGY.md`](WORKING_STRATEGY.md) step 1); starts all-`xfail`, greens case-by-case as the bridge covers DSL constructs.
3. Layer 4 before any default-flip; its overhead ratio + ADR-17 dual-run on real fixtures are the **quantitative cutover gate**.

## Open inputs
- The **ADR-01 per-model-class tolerance table** (`compatibility.md`) must be authored/located — it defines "no regression" numerically. Confirm where it lives or seed it from `baseline_vertical_slice`.
- Decision #8 (acceptable overhead ratio) sets the Layer-4 pass threshold.
