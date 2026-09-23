# Independent multi-output prediction contract (draft)

`by_source` followed by `merge: auto` trains one model per source. Its result is
**a set of named predictions**, not one prediction and not a mean of the
sources. Choosing the highest-scoring source for display does not discard the
other models or authorize an implicit choice during archive replay.

Here, an *output* is a prediction port attached to one source model. It is
independent of the number of target variables inside that port. For example,
two spectral sources predicting one concentration each yield two outputs,
each with one target column. A single source predicting three analytes yields
one output with three target columns. The two dimensions stay separate in
training results, archive metadata, and replay.

Concretely, for 20 samples and two spectral sources, `by_source` +
`merge: auto` produces **two** blocks of 20 predictions: one from the model
trained on source A and one from the model trained on source B. If each model
predicts three analytes, each block has shape `[20, 3]`; this is still two
outputs, not six. If an explicit `merge: mean` combines A and B, the graph
instead exposes one fused block of shape `[20, 3]`. An output ID identifies a
graph prediction port; a target name identifies a column within that port.

Training may rank the source models by their own validation scores, but every
score and prediction row must retain its output binding ID. The ranked winner
is a presentation/selection decision; it cannot change the stored graph's
output topology. A score for an explicit fusion node belongs to that fusion
output, not to either source model.

## Portable representation

The existing `PortablePredictorPackage.output_bindings` and replay request's
`output_binding_ids` already permit more than one output. A multi-output archive
must retain one binding per source, all fitted artifacts needed by those
bindings, and an attested topology descriptor mapping input source IDs to
output binding IDs. The descriptor is a proposed archive sidecar; this document
does not change the current package schema or claim that export is implemented.

```json
{
  "schema_id": "dag-ml.independent_outputs.v1",
  "kind": "independent_by_source",
  "sample_relation": "same_sample_ids",
  "outputs": [
    {"source_id": "source_0", "source_index": 0, "output_binding_id": "output:source_0"},
    {"source_id": "source_1", "source_index": 1, "output_binding_id": "output:source_1"}
  ]
}
```

The ordered `outputs` list must cover every source exactly once. Source IDs and
binding IDs must be unique; each binding must exist in the package, identify a
final-refit prediction port, and have a replayable artifact path through the
graph. The descriptor and the source-to-binding mapping must be included in the
archive's integrity/fingerprint validation. A missing or ambiguous mapping is
an export or load error, never a reason to select the first artifact.

## Replay and public result

Prediction input is a named set of source blocks (or a `SpectroDataset` with
the same stable source IDs) and explicit sample IDs. Every required source must
be present, with the saved feature schema and the same sample-ID set. Row order
may differ between sources: replay joins by sample ID, then returns every
output in the requested sample order. Missing/duplicate sample IDs, unknown
sources, incompatible feature schemas, or differing source sample sets fail
closed. There is no implicit missing-source policy in this first contract.

The result is keyed by `output_binding_id`; each block records `source_id`,
`sample_ids`, prediction kind, target/class metadata, values, and producer
provenance. A caller may explicitly request one output ID. A scalar convenience
accessor such as `y_pred` must reject an unselected multi-output result.
`merge: mean` remains a different, explicit graph operation that produces one
fused output binding. No archive reader may silently turn `merge: auto` into
`merge: mean` or a winner-takes-all predictor.

For the two-source example, the portable result has this logical shape (the
concrete wire format may differ by language):

```json
{
  "outputs": {
    "output:source_0": {
      "source_id": "source_0",
      "sample_ids": ["s1", "s2"],
      "target_names": ["analyte_a", "analyte_b", "analyte_c"],
      "values": [[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]]
    },
    "output:source_1": {
      "source_id": "source_1",
      "sample_ids": ["s1", "s2"],
      "target_names": ["analyte_a", "analyte_b", "analyte_c"],
      "values": [[0.9, 1.9, 2.9], [1.2, 2.2, 3.2]]
    }
  }
}
```

The core contract is language-neutral: Rust, CLI, C ABI, Python and future
bindings must preserve the same named output set, source/sample IDs, dimensions
and selection errors. A language binding may offer a convenience accessor only
when there is exactly one output or the caller names one. The binding must not
pick a winner or average blocks on its own.

For example, a future public API could expose
`result.outputs["output:source_0"]` and
`predict(model=archive, data=multi_source_data, output_id="output:source_0")`.
The exact Python method names are not fixed by this contract; the named output
and explicit selection behavior are.

## Acceptance tests before enabling export

1. Train on two genuinely different sources. Export and reload without a
   legacy refit; both output blocks must match their respective source-local
   refit predictions by sample ID through Python and the portable core/ABI.
2. Reorder input rows independently in each source and obtain the same
   sample-keyed values. Reject missing/duplicate IDs, a missing source, and
   incompatible feature axes or widths.
3. Verify archive fingerprints cover the topology and all required artifacts;
   reject a swapped binding, missing artifact, or duplicate source ID.
4. Verify explicit selection returns only that named output, while an
   unselected scalar accessor errors. Verify `merge: mean` separately against
   the mean-fusion graph oracle.

The current legacy `by_source`/`merge: auto` export is not an oracle for this
contract: it writes an archive but its `BundleLoader.predict` fails with
`No model step found in bundle` (and writes duplicate artifact names). The
nirs4all DAG-ML host archive now retains all fitted source models and exposes
named outputs with explicit selection. Its `BundleLoader` accepts independently
ordered source blocks with sample IDs and joins their rows using the DAG-ML
source-alignment primitive. It does not yet capture a signed
`PortablePredictorPackage` or validate every spectral axis descriptor. The
acceptance tests above therefore remain the gate for portable, cross-language
multi-output replay.
