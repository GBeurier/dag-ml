# Fold-local categorical targets, contract v1

Learning target bins independently within each CV training fold can give the
same integer class different meanings in different folds. Such predictions
cannot be pooled or voted across folds. The explicit `fold_local_macro` mode
scores every fold against that fold's transformed truth and ranks candidate
classifiers by the arithmetic mean of their validation scores. This is a
mean of scores, never a score of pooled predictions. The reduction is named
`macro`; existing `avg`/`w_avg` OOF semantics remain unchanged.

## Supported boundary

The first implementation supports one terminal mono-y classifier, sample-level
classification metrics (`accuracy`, `balanced_accuracy`, `f1`), complete
partition KFold training complements, a fixed ordinal vocabulary `0..k-1`,
and one full-training refit. Candidate generation may vary classifier
parameters while the target transformation recipe stays fixed. Quantile bins
that collapse the declared vocabulary are refused by the operator adapter.
An estimator may observe a subset of the declared bins in its training rows.
For a given fold, every candidate must attest the exact same fitted target
domain. A randomized/subsampled target operator that learns different mappings
for different model candidates is refused before selection; matching recipe
names and training IDs alone do not make these classification tasks comparable.

Joins, stacking, resampled/nested CV, target/group aggregation, HPO,
cross-domain probability/vote/ensemble reductions, and conformal calibration
are outside this mode and fail closed. Ordinary per-fold train/test scores
remain descriptive evidence. No `avg`/`w_avg` prediction or score is invented.
Standalone native portable predictor packages are explicitly refused until
their contract can retain the exact refit-domain closure. Attached native
replay and verified host archives are the supported inference paths.

## Declaration and attestation

The graph's `metadata.target_domain_policy` is validated as
[`target_domain_policy.v1.schema.json`](contracts/target_domain_policy.v1.schema.json):

```json
{"schema_version":1,"mode":"fold_local_macro","node_id":"classifier",
 "recipe_fingerprint":"<64 lowercase hex>","class_labels":[0.0,1.0]}
```

The recipe fingerprint is supplied by the operator owner and must stay
identical across candidates and folds. The owner records the recipe,
implementation versions and encoding in its provenance; the core does not
derive or implement a binning algorithm.

Each classifier result carries a `lineage.target_domain` following
[`target_domain.v1.schema.json`](contracts/target_domain.v1.schema.json).
It declares the recipe, the complete fitted `state_json`, its SHA-256 over
the exact UTF-8 bytes, the sorted unique `fit_sample_ids` and the ordinal
class vocabulary. The opaque state is a strict JSON object (bounded at
1 MiB, finite values, duplicate keys and runtime handles refused).

For FIT_CV/REFIT the coordinator requires exactly the IDs in its own
training views. Validation/test IDs cannot be substituted. Predictions and
scoring truth must contain one declared class ID per sample. The runtime
does not inspect operator internals: the fitted-state attestation is a
controller boundary claim, like other host fit-consumption attestations.

Every per-fold metric report retains the domain, including reports from
non-selected candidates. The native `macro` report has no single fitted
domain, so its optional `target_domain` is absent. Its member reports provide
the evidence. The standalone `TrainingOutcome` validator checks exact fold
coverage and training scope, recomputes the macro reduction, reconstructs
selection, and checks selected score/lineage/refit identity. Domain fields
participate in the enclosing outcome/package fingerprints. Older serialized
contracts without the optional fields keep their existing bytes.

## Inference

Attached native replay requires the exact source REFIT domain in the PREDICT
lineage. A verified host archive can compile a PREDICT-only graph by setting
`metadata.target_domain_refit` to the original captured REFIT domain alongside
the policy. The runtime requires byte-exact domain equality and refuses
training phases under this captured-domain declaration. Input inference
labels are unnecessary and cannot refit the domain.

Python adapters must check the explicit native capability
`dag_ml._dag_ml.fold_local_target_domains_schema_version() == 1` before using
the mode. A package version alone is insufficient: older runtimes can ignore
unknown graph metadata. There is no Python scoring or aggregation fallback.
