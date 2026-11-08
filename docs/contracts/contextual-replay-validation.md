# Contextual replay validation

Replay has three validation levels. Intrinsic validation (`validate`, `from_json`,
and Python `validate_training_replay_outcome_json`) checks structural contracts,
the portable explanation payload profile, and identifier uniqueness. Contextual
validation (`validate_against*`) also checks source/request bindings, complete
request/bundle data-key coverage, and selected explanation coordinates and targets.
Incomplete or unknown requested keys refuse even when the request and stored input
identities agree with each other. Neither level authenticates current rows.

The additive Rust current-cohort methods on `TrainingReplayOutcome` (source or
package) and `PortableRefitReplayOutcomeV3` inherit contextual key coverage, then
authenticate chosen relations against their recorded
digests, recompute input identities exactly (including `identity_fingerprint`),
and check bound output membership. Executors use these same methods. An incomplete
or unknown requested key is `ReplayCurrentCohortError::Invalid`; an otherwise valid
request with a missing supplied envelope is `Missing(EnvelopeAbsent)`. A binding
without any closure data key yields `Missing(NoClosureDataKey)` only after all
other checks succeed. Executors tolerate this last result without making a row
claim for that binding; stored callers retain its explicit unverified result.
These errors are Rust-only, with no new serialized fields or binding API.

For each selected binding, membership uses only data keys in its predictor
closure. A V2 `predict_cohort` selects its own relations; otherwise the coordinator
relations apply. Development and current relations are never combined. Mixed
V1/V2 keys each contribute their own authenticated relations. Prediction rows use
physical `sample_id`; observation predictions use `observation_id`; aggregates
use sample, non-null target, or group IDs. A `RelationMetadata` grouping key uses
only nonblank string values at that key, instead of `group_id`. Relation unit
overrides, origin IDs and composed source IDs do not authorize these typed rows.
Excluded records remain eligible. Partial and reordered outputs remain valid:
this is membership, not complete row coverage. IDs are unique within and across
blocks of each family for a binding; distinct bindings may share IDs.

Current-row validation does **not** prove that host numerical buffers match
`data_content_fingerprint`. In EXPLAIN, bound prediction rows are checked, but
rows behind explanation payloads remain **unverified**: the scheduler provides
no EXPLAIN cohort view. No numerical or attribution validity is asserted.

Every callback explanation first passes producer/port and payload validation,
including explanations from unselected nodes. Collection then keeps selected
(node, resolved port) coordinates only. A named target must belong to that
binding; `None` remains global. Class aliases are not target names. Ambiguous
coordinates anywhere in the saved source/package refuse. Stored contextual
validation refuses unselected explanations and an empty EXPLAIN selection.
Executed lineage remains truthful: it includes the selected closure, stays
inside the plan, and has one record per executed node. Legacy omitted ports use
the existing unambiguous port resolution; required V3 ports are a separate gap.

The shared explanation payload profile refuses runtime-handle markers and the
recursive ASCII case-insensitive keys `raw_features`, `feature_matrix`,
`raw_spectra`, and `raw_wavelengths`. A matrix under an arbitrary key such as `X`
remains legal. This is a portable payload profile, not a general raw-data detector
or a mathematical payload validator.

The normal WASM runtime controller and C ABI decode raw NodeResult JSON with
the existing strict external-contract decoder. The CLI decodes the one-shot raw
NodeResult and each whole persistent control_frames_v1 response line with that
decoder. Python in-process callbacks provide structured mappings instead of raw
JSON, so duplicate raw tokens cannot be observed at that boundary. Integer tokens must fit the signed
or unsigned 64-bit domains: `u64::MAX` and `i64::MIN` remain valid; larger integer
tokens refuse. Finite exponent-form binary64 values remain valid. JavaScript
`JSON.stringify(2e19)` emits the integer token `20000000000000000000`: this was
accepted before and is now refused by the strict profile. Hosts must supply a
valid numeric token representation; no new serializer API is provided here.
HPO optimizer replies, complete-worker results and the existing fold-result
path also use the strict decoder. Other WASM request, envelope and checkpoint
decoders are separate; this does not make all WASM ingress strict.

Envelope-wide/dagml-data parity, intermediate unbound predictions, public binding
exposure of the additive APIs, and explanation cohort views remain separately
scoped. The synthetic host fixtures test generic coordination, without native,
IO, scientific or installed-package qualification. Scientific profiles remain
exactly `native-workflow@1` and `native-conformal@1`.
