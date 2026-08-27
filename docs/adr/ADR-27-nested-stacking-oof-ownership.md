# ADR-27: Nested stacking OOF ownership

**Status**: proposed (2026-08-27)
**Supersedes**: none. It narrows the stacking part of ADR-22 without changing
the OOF safety invariants in ADR-05.
**Blocks**: R2 `API-001` stacking, the three duplication/merge cases in the
compatibility ledger, and the native-default flip.

## Context

The current scheduler executes every node for one outer `FIT_CV` fold. For a
stacking graph this lets a meta-model receive base predictions for that outer
fold's validation rows, fit on those rows, then score those same rows. Filling
missing rows at `REFIT` time cannot correct that leakage or reproduce the
legacy nested stacking semantics.

Stacking needs two distinct prediction roles: base-model OOF evidence used to
fit the meta-model, and outer-fold predictions used only to assess the complete
stack. They must never be represented by the same cache requirement or fold
identity.

## Decision

1. A native stacking campaign is nested. For each attested outer fold, the
   scheduler derives a deterministic inner fold set over exactly that outer
   fold's training identities. It runs each base branch across the inner folds,
   joins complete sample-level inner OOF predictions by stable sample ID and
   port, fits the meta-model on that joined matrix, and only then predicts the
   outer validation identities.
2. The outer validation identities are evaluation-only. They are never inputs
   to a meta-model `FIT_CV` operation, even temporarily, and no imputation,
   averaging or cache fallback may manufacture a meta fitting row for them.
3. OOF cache requirements and prediction lineage carry an explicit stacking
   layer plus parent outer-fold identity. Inner and outer records are distinct
   resources. A cache is valid only when its declared universe, fold set,
   source port, prediction level and parent scope all match exactly.
4. `REFIT` has the same separation: base branches create a complete OOF matrix
   over the selected fitting cohort under a deterministic refit-inner fold
   set; the meta-model fits on that matrix; final/inference inputs use only
   off-fold base predictions. Partial OOF coverage is a typed refusal. Mean
   imputation is not an R2 stacking completion path.
5. The operation is scheduler-owned. Python may lower an eligible graph and
   register process-local operators, but may not run an independent CV loop or
   assemble meta features outside the attested DAG-ML campaign. The persisted
   outcome/package records both fold-set fingerprints, OOF requirements and
   the selected meta-model lineage.
6. The proof gate compares the nested native campaign with the legacy oracle
   for the supported duplication/merge forms, includes leakage and
   parent-fold/cache-transplant negatives, then process-separates package
   replay. A case remains `supported` in the compatibility ledger only after
   this gate passes without legacy fallback.

## Consequences

- The existing single-scope `FIT_CV` loop is insufficient for stacking and
  needs a dedicated nested campaign path, not a relaxed OOF-refit policy.
- Wire, cache and bundle schema evolution must be additive and dual-readable;
  no old OOF record may be reinterpreted as nested evidence.
- Until the nested path and proof gate exist, stacking remains a visible
  expected fallback and `DEFAULT_ENGINE` cannot flip to native.

## Blocks

Nested scheduler phase/resource design, cache/bundle lineage extensions,
process-adapter data views for derived inner folds, native parity fixtures and
the three stacking entries in the R2 compatibility ledger.
