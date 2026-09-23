# Capture seam for independent source outputs

This note narrows the implementation work behind
[`MULTI_OUTPUT_PREDICTOR_CONTRACT.md`](MULTI_OUTPUT_PREDICTOR_CONTRACT.md).
The contract concerns prediction ports, not the number of target columns.

## Current evidence boundary

The core already accepts multiple `TrainingOptions.outputs`. A completed
`TrainingOutcome` can build a `PortablePredictorPackage` containing every
`OutputBinding` and the refit artifact closure. A `TrainingReplayRequest` names
the binding IDs to replay, and PREDICT validation requires exactly those
outputs. `PortablePredictorPackage::select_output` requires an explicit ID.
`portable_package_replays_all_named_outputs_and_only_explicit_selection` tests
this path with two different prediction ports and checks sample IDs.

The nirs4all `by_source`/`merge: auto` paths currently bypass that training
operation. The CV path in `pipeline/dagml/run_paths.py` constructs a DSL and
calls `run_cv_refit_bundle`; the no-splitter path in `pipeline/dagml/full_train.py`
calls `execute_phase_in_process` or the matching CLI command. Both paths retain
the returned `ScoreSet`, `NodeResult` frames and host refit estimators in
`RunResult`. `pipeline/dagml/native_results.py` persists those pieces and
artifact hashes, but it does not persist a signed `TrainingRequest` or a
`TrainingOutcome`. The core package cannot be reconstructed from a score set
and artifact list: the effective plan, bundle, output bindings, training
influence, data identities, and their cross-linked fingerprints would have to
be invented after execution.

The current `.n4a` host archive is a useful Python replay profile: it stores
all source estimators and a source-to-output descriptor. It can split one
concatenated matrix by saved feature widths or accept named source blocks with
sample IDs. For named blocks, `align_named_source_rows` in DAG-ML core validates
exact source/sample coverage and returns row permutations; Python applies
those permutations to its host-owned feature buffers. The primitive is exposed
through Python, C and WASM. This host archive is still not a
`PortablePredictorPackage`: it lacks the signed training outcome, portable
artifact bindings and attested source-to-output relation.

## Minimum honest CV capture

1. Compile the existing by-source graph and fold campaign into one signed
   `TrainingRequest`. Declare one `TrainingOptions.outputs` entry per terminal
   source model, with a stable `output_binding_id` and its actual node/port.
   Keep the score-ranking `selection_output_id` explicit; it must not remove
   any independently declared output.
2. Bind each model input to its named source in the campaign data requirements.
   Capture `TrainingDataIdentity` for every requirement, with the source-local
   schema/content fingerprint and common sample relation. The source ID to
   output binding map must be attested by the resulting package/archive, not
   inferred from array position when loading it.
3. Execute the request through `execute_training` with the existing host
   operator callback. Preserve the returned `TrainingOutcome` and call its
   `to_portable_predictor_package` method. Store all host-owned joblib models
   as sidecars addressed by the package's `PackageArtifactBinding` records.
   A missing source artifact or output path is an export error.
4. At replay, build named source envelopes with stable sample IDs. Validate
   source names and feature schemas, reject missing or duplicate sample IDs,
   align each source by sample ID through the native relation contract, and
   send every requested `output_binding_id` in one `TrainingReplayRequest`.
   Return blocks keyed by binding ID. A scalar accessor requires one selected
   binding; the package's selection metric is never a replay default.

The no-splitter route needs its own attested package construction if
`execute_training` requires FIT_CV evidence for selection. It must preserve
the current full-train semantics and must not manufacture CV scores or folds.
The existing portable refit package contract is a candidate for that route;
it should be tested before replacing the phase runner.

## Executable CV seam and remaining production blockers

`nirs4all.pipeline.dagml.attested_by_source.execute_attested_by_source_cv`
now builds a signed request before execution, calls `dag_ml.execute_training`,
and obtains one genuine `TrainingOutcome` and `PortablePredictorPackage` with
all source output bindings and refit artifact records. The integration test
uses three source-local Ridge models on a real CV dataset. The public
in-process runner now calls this operation for one-refit campaigns without
operator generators. DAG-ML builds its `ScoreSet` after REFIT and retains the
selected run's final/test reports alongside selection's validation reports.
The public projection has the same 18 reports and prediction rows as the CLI
runner (three validation folds, validation average, final train and external
test for each source); an integration test compares every predicted block
against CLI. The CLI route, top-k refits and operator-generator campaigns
still use the existing runner.

The prototype currently gives each model requirement the same signed
multi-source data-plan envelope. Attempting distinct source-local content
fingerprints under that shared external plan fails with
`duplicate external data-plan envelope with different payload`. To attest
each source independently, the compiler must mint distinct plan identities
and bindings for the source views before training; changing fingerprints after
execution is not acceptable. The host Python/joblib artifact sidecars also
need an exact package artifact-binding map before claiming a cross-language
portable archive.

## Gate before claiming portable parity

- Train two genuinely different source models and export one package with two
  final-refit bindings and two artifact sidecars. Load and replay both through
  Python and the core/ABI; their values must match each source's refit oracle.
- Independently permute source rows while preserving sample IDs. Results must
  be equal by sample ID. Reject missing/duplicate IDs, absent sources and
  changed source feature schemas before invoking a model.
- Tamper with the source-to-binding map or remove/swap an artifact; package and
  archive validation must fail before deserializing host models.
- Test explicit one-output selection and all-output replay separately. An
  unselected scalar call must fail; no score winner or implicit mean may be
  used as the default.
