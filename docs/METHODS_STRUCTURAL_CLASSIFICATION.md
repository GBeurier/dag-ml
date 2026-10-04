# Native Methods structural classification

This additive profile runs one serial, mono-target classification campaign in
the Python native host. It supports a four-source early classifier or learned
late fusion with two to four distinct single-source branches. Every head is the
actual Methods `models.classification.pls_logistic` estimator. Supported native
selection metrics are accuracy, balanced accuracy and support-weighted F1.
Classification parallel execution, GPU profiles, probability-dependent selection
metrics and unimplemented DAG R/WASM/Octave classifier owners are refused.

The raw operator is `N4mMultimodalClassifierPipeline`; the terminal late operator
is `N4mRoleClassifierPipeline`. Both carry a closed `classification` declaration:

```json
{"schema_version":1,"class_labels":[0,1,2],"label_names":[-17,42,901]}
```

The sorted unique original labels are homogeneous int64 values or UTF-8 strings,
including an empty string. They are discovered from Train only. The graph's
`metadata.classification_targets` repeats that exact typed vocabulary and adds
`sample_labels`, mapping precisely the signed Train sample IDs to internal
contiguous IDs. Test labels may be scored against this table but cannot discover
or enlarge it. Unknown labels, mixed types and absent classes in any actual
native training scope are refused before optimizer or operator callbacks.

The native preflight derives grouped outer scopes, nested grouped inner scopes
and the independent full-Train REFIT inner OOF scopes from the existing planner.
It checks classifier component counts, encoder PCA capacity and the closed
16,777,216-cell bounds for one-hot and ordered OOF matrices. The mixed metadata
encoder guarantees at least two encoded columns: numeric plus a nonempty native
categorical vocabulary. No host constructs folds or pads probability columns.
The same limit bounds the actual logistic design `rows × (components + 1)`
and Hessian `((classes - 1) × (components + 1))²`, with overflow refusal.
Native search admission checks conservative signed component-domain maxima and
forced counts before optimizer callbacks; metric changes cannot admit parallel
classifier operators in either a catalogue or the effective plan.

Raw recipes preserve the existing closed scaler, tensor PCA and mixed-column
encoder declarations and all four raw source schemas. The head is
`{"method_id":"models.classification.pls_logistic","params":{"n_components":1,"max_iter":200}}`.
The late operator uses exactly one corresponding `steps` entry and signs its
ordered `source_order`. Effective numeric parameters are positive integers:
raw `model__n_components`/`model__max_iter`, meta `n_components`/`max_iter`.

Each classifier emits `y_hat` (one internal class ID per sample) and
`probabilities` (one complete normalized distribution in the declared class
order). Only `y_hat` is scored and deployed. A DSL late `merge_model` declares
`source_ports={logicalProducerId:"probabilities"}` and `requires_oof` edges;
generator expansion remaps the source IDs and port-map keys together. Probability
column names are `class:0`, `class:1`, etc. Meta feature names flatten declared
branch order followed by class order as `<generatedProducerId>/class:<ID>`.
Class-column permutations and foreign, Train or Test fitting inputs are refused.

The deployment request explicitly selects `y_hat`, `class_label`, sample level,
target `y` and `class_labels=[["0","1","2"]]`. This describes the numeric IDs
actually emitted; original typed names remain signed separately and the SDK
decodes them only for the public result. Existing regression output validation
and byte contracts are unchanged.

Use `dag_ml.multimodal_classification.ClassificationTopologyController` with
the same raw source/target/edge boundary as `MethodsTopologyController`.
Raw ownership is `controller:methods.python.multimodal.classification`, and meta
ownership is `controller:methods.python.classification`. Target-free consumers
use `allow_fit=False`, no targets and the exact selected `node_params` map.
The facade splits this complete map by owner after refusing unknown nodes.

Portable RAW wrappers retain exact graph, selected recipe, parameter fingerprint,
typed vocabulary and branch/class feature order. Raw states are N4MC format 1;
meta states are N4ME format 1. New classifier writers require ABI 2.17.0.
Independent Rust and Python validators check bounds, nested state checksums,
canonical recipe bytes, encoder state widths, the class table, head capabilities,
parameters and embedded PLS state structure. Methods owns numerical hydration.

Archive V2 retains `multimodal_pipelines` and `role_pipelines` inventory keys and
adds exact classifier kind/profile pairs, without changing historical regression
fixtures. A late REFIT keeps genuine foldless Test output when present, or the
existing strictly validated artifact-only deployment when no Test exists. It
never fabricates training Final predictions from OOF features. Fresh PREDICT
hydrates selected states and emits genuine Final labels and probabilities with
no training targets or fit call.

Qualification must run fresh native artifacts after whole-phase review. Authored
wire/host witnesses are not scientific qualification; the SDK's grouped
independent numerical oracles cover the complete early/late campaigns, export
and fresh-process replay.
