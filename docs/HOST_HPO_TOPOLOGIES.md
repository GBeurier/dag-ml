# Conditional early and learned late topology HPO

The additive V2 catalogue is prepared through Python
`prepare_host_hpo_topology_catalogue(dsl, envelope, controller_manifests,
parameter_bindings, scored_nodes, selector_path="__recipe__")`.
Preparation compiles declarations and validates identity; it does not fit an
operator or ask an optimizer. Existing V1 catalogue JSON, fingerprints and
global generator content labels retain their original contracts.

Declare one native operator generator. Each alternative is a complete sequence:
one early model, or a duplication branch containing named source models followed
by a `merge_model` meta-model. Native compatibility syntax also accepts `_or_`
arrays of sequences, with `merge="predictions"` immediately followed by a model
lowered to `merge_model`. Structural choice enumeration remains compiler-owned.

`parameter_bindings` maps a public numeric axis to a list of logical source DSL
destinations, each `{node_id, param_path}`. `scored_nodes` lists the explicit union
of logical sink IDs. The compiler's exact node minting function resolves these
IDs separately for every recipe; hosts must not reconstruct generated IDs.
Each recipe has exactly one active terminal FIT_CV scored model. Every active
model must reach that sink. Destinations must be declared finite numeric model
parameters, distinct and used in at least one recipe. A public axis has at most
one active destination. Proposals require every active axis and reject inactive,
unknown and nonnumeric patches before operator callbacks.

Late meta-models declare ordered `sources`, `include_original_data=false`,
metadata `stacking_oof_execution="nested_oof_v1"` and
`stacking_refit_oof="partitioned_inner_v1"`, plus explicit grouped `inner_cv`.
Prediction-only meta nodes have no raw data bindings. Each raw source model
retains the complete raw source contract even when its Methods recipe selects
one source. The native scheduler constructs inner OOF inside every outer train
scope and independent full-training OOF before REFIT. Training lineage retains
all report-grade outer, parent-bound inner, and requested full-REFIT inner FIT_CV
runs, with exact coverage of the signed planner coordinates. Tensor PCA capacity is
checked against raw width and every actual native training scope before fitting.

The returned schema-2 catalogue carries `topology_contract`. Its per-recipe label
signs the full active graph, campaign, bindings and sink, including nested branch
contents/order and CV policies. Consumers recompile this declaration and compare
all derived entries before execution; resealing a changed derived graph, binding
or label is insufficient. SELECT and the existing winner resolver retain the
selected native VariantPlan, apply only its active node parameters and prune
inactive data contracts. V2 winner templates may not patch inactive nodes.

The current feature runs serially and rejects progressive pruning and parallel
worker windows. Preparation is exposed through Rust and Python; CLI execution
uses the ordinary serialized host-HPO request. No dedicated preparation C/WASM
endpoint is added. Native predictor archive transport combines complete N4MF
raw-source wrappers with the existing N4ME role-pipeline wrapper for meta Ridge;
the Python role wrapper requires both exact Python controller owners. Bare
generic N4ME estimator transport is outside this archive contract.

Deployment output bindings retain `prediction_source="final_refit"`, which
identifies the fitted predictor rather than relabeling a prediction partition.
Existing Final blocks retain their original payload. Only when an OOF-consuming
model emits no Final blocks may a completed REFIT bind genuine foldless Test
rows, together with `refit_test_cohort` proving their exact external-test sample
identities, targets and disjointness. This fallback is currently sample-level.
Without a prediction cohort, the same fitted learner may retain an explicit
`artifact_only=true` binding with empty prediction arrays. Both fallbacks require
the selected producer's exact retained fitted artifact and REFIT lineage;
artifact-only outputs may not conceal retained Test reports. CV/SELECT evidence,
prediction caches and predictor closure remain mandatory. Neither field appears
in legacy Final outputs or PREDICT replay results, which still require real
Final prediction blocks. No training predictions are fabricated from OOF rows.
