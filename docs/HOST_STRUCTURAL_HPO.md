# Host HPO over existing operator generators

Structural host HPO is additive to fixed-topology host search. The coordinator
compiles the declared operator generator into an immutable catalogue, prunes
each proposed recipe with the existing operator planner, and scores its own
validation reports. The first public SDK profile is dense mono-source grouped
regression with optional StandardScaler and Ridge or PLSRegression(scale=False).
Methods continues to own ask/tell and conditional search-space membership.

```python
catalogue = dag_ml.prepare_host_hpo_structural_catalogue(
    dsl, envelope, manifests,
    {"model.alpha": "alpha", "model.n_components": "n_components"},
)
request["target_node"] = catalogue["entries"][0]["target_node"]
request["parameter_bindings"] = {}
request["structural_catalogue"] = catalogue
search = dag_ml.run_host_hpo_search_in_process(
    dsl, envelope, manifests, request, operator_callback, optimizer_callback,
    candidate_callback_factory=callback_factory,
)
selected_request = dag_ml.resolve_host_hpo_structural_winner(
    request, search, ordinary_unsigned_training_template,
)
result = dag_ml.execute_training(
    selected_request, selected_data_envelopes, relations,
    selected_training_influence, selected_operator_callback,
    outcome_id=outcome_id, run_id=run_id, bundle_id=bundle_id,
    artifact_callback=artifact_callback,
)
```

The caller constructs ordinary W1 data/output/resource declarations using its
existing training integration. It derives envelopes and influence for the
returned selected graph before executing that signed request. `execute_training`
executes the sole winning recipe's CV/SELECT/REFIT and retains its real fitted
artifacts; export and fresh replay use the existing training contracts.

The original two declaration sites lower to native cartesian generator stages:
one raw/transform stage and one model stage. The host never enumerates the
cartesian product or assigns recipe IDs. Catalogue entries expose the native
VariantPlan, content label, pruned graph, selected model, graph/controller hashes
and numerical bindings. A parameter is active only on recipes whose declared
model owns that local parameter. The selector defaults to `__recipe__`; it is an
internal categorical optimizer axis, outside public path normalization.

An ask response contains that exact recipe ID and every active numerical path.
Unknown recipes, inactive paths, missing active values and nonfinite/non-numeric
values fail before candidate factories or operator callbacks. A callback factory
has the existing signature `factory(trial_index) -> callback`; its callback binds
the catalogue graph for the recorded selector and receives ordinary native tasks.
Task IDs retain `host_hpo:trial:0000000000` form. Their VariantPlan choices retain
the original recipe choices plus a host-HPO choice that binds recipe ID/catalogue
hash and effective overrides. Final training preserves the original native recipe
VariantPlan and records the winning trial and parameters in campaign metadata.

Core independently recompiles and compares every catalogue entry before any
candidate execution. Worker tasks additionally reconstruct their selected plan
before invoking controllers. Each trial has its own existing controller/provider
lifecycle; pruned graphs cannot consume inactive transformer handles. Native
ScoreSets and the existing selection policy choose the winner, including every
required fold under an explicit fold reduction.

Checkpoints remain schema 1. A structural checkpoint adds the optional
`binding.structural_catalogue_fingerprint`; its objective hash also binds the
entire ordered catalogue, declarations, activation and optimizer descriptor.
Resume validates the native seal, catalogue, graph/controllers/data/folds and
historical native scores before asking or invoking callbacks. Total trial budget
may increase under the existing rules. Fixed-topology requests omit every new
field, preserving their canonical bytes, objective hashes and checkpoint seals.
No conversion between fixed and structural checkpoints is performed.

The closed schemas are `host_hpo_search_request.schema.json`,
`host_hpo_structural_catalogue.schema.json` and `host_hpo_checkpoint.schema.json`.
The catalogue is bounded to 4096 recipes and one existing native cartesian
operator generator with shared explicit folds. It does not add a new generator
language, optimizer, numerical implementation or Methods ABI.
