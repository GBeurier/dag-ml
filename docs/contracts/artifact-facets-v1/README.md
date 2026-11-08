# DAG owner artifact facets v1

`dag_ml_core::artifact_facets` supplies scientific owner declarations and
validation independently of ZIP, Core, IO and host loaders. The machine-readable
wire schema is [artifact_facets.v1.schema.json](../artifact_facets.v1.schema.json).
No binding or ABI surface is added.

`ArtifactFacetGraph` serializes as `{artifacts: [...], links: [...]}`. Each
artifact has exactly `id`, `kind`, `owner`, `content_type`, `identity`, `storage`
and `requirements`. Embedded and reference artifacts both require the same full
`OwnerIdentity`: `owner`, `schema`, positive `schema_version`,
`fingerprint_profile`, `fingerprint`. `owner` must agree with the artifact owner.
`content_type` is `{id, version}`. `requirements` must be explicitly null or
`{"artifact_id": "declared ID"}`, with no extra fields. Both the public serde
decoder and `from_json` reject an omitted `requirements` field. Generic
validation checks reference shape and ID existence only; the owner validates
requirement meaning. Identity `schema`/`schema_version` are structurally
independent of `content_type`; current owner proof matching remains exact.

Embedded storage is `{kind: "embedded", root_member, member_paths}`. The root
occurs in the nonempty list; each physical member belongs to exactly one facet.
Reference storage is `{kind: "reference", root_sha256?}` with an optional
lowercase SHA-256 string of the root bytes. Omission is allowed; explicit null,
uppercase or malformed digests are refused, including by public serde. References are unresolved context, not
missing ZIP bodies. No network retrieval occurs. Raw SHA and semantic identity
are separate, including when the native owner's particular profile hashes bytes.
Core remains responsible for safe member paths, container digests, limits,
entrypoint closure and binding any admission to a specific container SHA,
entrypoint ID, root/input IDs and content types.

The closed artifact kinds are `recipe`, `dataset`, `model`, `run`, `result`,
`score`, `prediction`, `residual`, `training_evidence`, `full_refit`, `provenance`
and `attachment`. Calibration is deferred. Neither a kind nor an archive-declared
boolean authorizes an operation.

## Owner byte proofs and profile domains

`validate_owner_facet_bytes(&FacetContentType, &[u8])` returns
`FacetResult<Option<ValidatedOwnerArtifact>>`. Known malformed content or a wrong
version is refused. An unknown owner content type returns `None`; its bytes
remain opaque and cannot acquire DAG semantic proof. `ValidatedOwnerArtifact`
has private fields, no `Deserialize` or `Serialize`, and retains original bytes,
raw SHA, validated typed payload and derived identity. Getters expose its current
validated outcome/predictor/refit objects for composition. It is not an
authenticity signature or an assertion to serialize into a container. These
owner proofs establish consistency; they do not authenticate producer identity.

Known owner content IDs and versions:

| Content ID | Versions | Fingerprint profile |
| --- | --- | --- |
| `dagml.graph_spec` | 1 | `dagml.facet.graph-spec.tcv1.v1` |
| `dagml.execution_bundle` | 1, 2 | `dagml.facet.execution-bundle.tcv1.v1` |
| `dagml.score_set` | 1, 2 | `dagml.facet.score-set.tcv1.v1` |
| `dagml.training_outcome` | 1, 2 | `dagml_tcv1`, excluding `outcome_fingerprint` |
| `dagml.portable_predictor_package` | 1, 2 | `dagml_tcv1`, excluding `package_fingerprint` |
| `dagml.initial_full_refit_package` | 1 | `dagml_tcv1`, excluding `package_fingerprint` |
| `dagml.portable_refit_package` | 3 | `dagml_tcv1`, excluding `package_fingerprint` |
| `dagml.prediction_cache_payload_set` | 1, 2 | `dagml.facet.prediction-cache-payload-set.tcv1.v1` |
| `dagml.execution_plan` | 1 | `dagml.facet.execution-plan.tcv1.v1` |

The identity schema is the content ID, and its version is the validated owner
version. The accepted profile list is `DAG_FACET_FINGERPRINT_PROFILES`. For each
of the five new profiles the preimage is **TCV1 of this object**, including the
ordinary TCV1 prefix:

```text
{
  "profile": "<exact profile ID>",
  "content_type": {"id": "<exact content ID>", "version": <integer>},
  "payload": <validated typed owner projection>
}
```

For these five profiles only, the validated Rust owner object is serialized
with `serde_json::to_vec` and parsed directly with `parse_typed_json_bytes` into
the domain. This normalizes owner-defined optional/default fields: equivalent
omitted defaults or explicit nulls have the same identity as the current owner
projection. Original bytes independently pass strict token preflight **before**
owner decoding, then remain retained exactly, including whitespace and token
spelling. Fingerprinting introduces no intermediate `serde_json::Value` pass or
JavaScript binary64 conversion. Metadata integer versus binary64 kinds and the
i64/u64 domains remain distinct; `u64::MAX` stays exact. Duplicate keys at any
depth, NFC-colliding keys, non-finite/out-of-range numeric tokens, unknown typed
fields and positional struct arrays are refused.

The existing actual `dagml_tcv1` self-fingerprint profiles still hash original
typed bytes excluding their self-fingerprint field, without this normalization. Existing
`TrainingOutcome::from_json` and package parsers continue to verify their stored
self-fingerprints against original TCV1 and current owner validation. A cache
subtree's alternate integer/default/null representation can be a valid
standalone cache payload while an outcome containing that altered subtree is
rejected by the outcome's current self-fingerprint checks. Facet normalization
does not make that outcome valid or replace its original bytes.

Existing GraphSpec/plan historical hashes and internal whole-bundle TCV1 hashes
are preserved inside owner payloads. Existing archive profile references are
unchanged. No ScoreSet or ExecutionBundle self-fingerprint field is invented.
The recipe is the **actual compiled GraphSpec**, not the original DSL. There is
no claim that their identities are interchangeable.

## Relations and refusals

Every link contains `id`, `owner`, `relation`, `source_artifact`, `target_artifact`.
The DAG relation owner is `dag-ml`. Unknown relation strings, reversed directions,
self-links, duplicate semantic edges, duplicate artifact/link IDs, undeclared
endpoints and invalid requirement references are refused. No arbitrary
one-dataset-per-run cardinality is imposed; bindings and cohorts can be multiple.

| Relation | Source | Target |
| --- | --- | --- |
| `recipe_of` | recipe | run, result, training_evidence, model, full_refit |
| `dataset_of` | dataset | run, result, training_evidence, model, prediction, residual, score, full_refit |
| `produced_by` | model, result, training_evidence, prediction, residual, score, full_refit | run |
| `evaluates` | score | prediction, result, training_evidence |
| `residual_of` | residual | prediction |
| `evidence_for` | training_evidence | model |
| `refit_of` | full_refit | training_evidence |

`inspect_artifact_facet_links` returns a per-link verdict of `validated`,
`unverified` or `mismatch`, with a reason. `validate_artifact_facet_links` refuses
any mismatch. Any supplied proof that disagrees with the declared full identity
or optional raw SHA is immediately refused, even in diagnostic inspection. A
known mismatch never becomes unverified. Declarations alone supply no proof.

Known owner payloads must have a compatible scientific kind before any link
validation. Unknown owner contents remain structurally opaque/unverified.

| Validated payload | Compatible scientific kinds |
| --- | --- |
| GraphSpec | recipe |
| ExecutionBundle | run |
| ScoreSet | score |
| TrainingOutcome | result, training_evidence, run |
| PortablePredictorPackage | model |
| InitialFullRefitPackage | full_refit, model, run |
| PortableRefitPackageV3 | full_refit, run |
| Prediction-cache payload set | prediction |
| ExecutionPlan | attachment (storage only) |

Known contents can also be stored as attachment/provenance. These kinds cannot
supply scientific relations or `score-inspect` admission. A ScoreSet relabeled
model, for example, is an immediate known mismatch rather than unverified.

Recipe checks compare current validated graph identities with owner graph
payloads. For a standalone bundle, its current owner historical graph binding
is checked directly against the validated GraphSpec. Cache-to-bundle checks call
`BundlePredictionCachePayloadSet::validate_against_bundle` on that exact pair;
matching bundle closure still cannot establish a producer run label.
Outcome/package evidence checks reuse current validators and
compare selection, plan, output/data identities and fitted-state references.
A training run facet references the complete owner-fingerprinted outcome identity, which
binds its actual run label. ExecutionBundle is supporting closure with its own
new facet profile; because it has no run label, a bundle alone cannot validate
a producer-run claim. Initial/refit run references identify their complete
validated refit package. No independently fingerprinted run record is invented.

`evaluates` detects direct score/plan mismatches against its actual validated
outcome target, but remains unverified:
this slice does not establish the complete sample population, target/class and
evaluation scope from ScoreSet. No missing fields are invented. `residual_of`
remains unverified without a complete residual owner contract. Evaluation against
caches has no unlinked outcome witness scan: unrelated proofs cannot invent a
validation or mismatch. Refits obtain a source-package witness only through a
declared `evidence_for` link from their parent evidence, matching the exact
source-package self-fingerprint. Under the fixed validated PPC schema/profile,
that fingerprint already determines the full package identity, including ID and
version (absent a hash collision); no fabricated same-hash/different-identity
case is accepted as test evidence. A different valid package export ID has a
different fingerprint and cannot replace the child's exact source. Conflicting
declared evidence is checked by the actual `evidence_for` owner validation.
BTreeMap key order
and unrelated context additions have no scientific authority.

`dataset_of` remains unverified unless an explicit, trusted
`FacetDatasetBindingValidator` composition port validates actual IO content
identity **and** its IO-to-DMD/DAG envelope/binding projection for the target.
Use `context.with_dataset_binding_validator(&live_port)` to attach an authority
without cloning the retained bytes or requiring a static provider lifetime.
`external_bytes` contains supplied candidate bytes, not owner validation. Entries
for undeclared IDs are refused. For the known generic profile `sha256.raw.v1`,
the SHA of these bytes must match the declared identity before any port runs,
for both embedded and reference facets. An optional reference root SHA is also
checked. This initial dataset facet profile identifies the exact normalized
IO-emitted bytes only: it is not a logical dataset identity, row-order invariant,
scientific parity claim or IO-to-training mapping. Unknown scientific identity
profiles remain the explicitly trusted owner's responsibility.

The port receives the supplied candidate dataset bytes and a live target proof;
it must validate actual IO content/profile plus the real IO-to-DMD/DAG mapping,
not just compare an IO fingerprint to `TrainingDataIdentity`, which
hashes a different object. Its mismatch verdict refuses validation. This is a
caller-selected semantic authority, never an implementation selected by archive
JSON. There is no mandatory IO dependency, data-buffer materialization or built-in
IO parser. The default fixtures do not qualify IO dataset identity.

## Direct builders, states and admission

`build_training_artifact_facets(outcome, package)` validates both current objects
and their exact cross-link. Call it immediately after `execute_training` and
`TrainingOutcome::to_portable_predictor_package`, before any archive assembly.
`build_training_artifact_facets_from_bytes` retains supplied outcome/package
bytes exactly. The returned `ArtifactFacetExport` contains `graph`, exact
`members: BTreeMap<String, Vec<u8>>` and a process-local proof `context`.
`validate()` checks exact member coverage, original root bytes, supporting raw
state bytes and semantic mismatch refusal.

The training export has recipe/GraphSpec, result/outcome, score/ScoreSet,
model/predictor-package and supporting execution-bundle, plan and optional
prediction-cache members. Run and training evidence are references to the same
owner-fingerprinted outcome identity, resolved in the live context. It does not claim the result member a
second time. Model state members use content-addressed paths
`dagml/facets/<SHA256(facet ID UTF-8)>/states/<raw SHA256>.bin`. Equal raw bytes
share one path within a facet; distinct facets have distinct namespaces. The
owner-fingerprinted package's original URIs, ArtifactIds and embedded state
closure remain unchanged. `expected_facet_state_members(facet_id, proof)` derives
ArtifactId-to-path/size/digest inventory from the actual validated root raw map.
This inventory describes external physical closure for Model/FullRefit only.
Its ID/proof arguments cannot establish the actual facet kind. Readers must call
`validate_facet_members` for descriptor/proof agreement and the actual kind's
root-only or state-closure policy; inventory alone never authorizes admission.
`validate_facet_state_members(artifact, proof, members)` verifies exact root bytes,
membership and external state bytes against that root. A later Core bridge must
perform this owner-root comparison rather than trust a stored inventory. No
separate Methods facet duplicates ownership of the package's native children.
No owner-fingerprinted JSON is stripped or rewritten to remove raw state duplication inside
its own contract. External consumers can resolve the reference from exact owner
bytes. No legacy archive builder or reader is called.

`build_initial_refit_artifact_facets` emits a full-refit package, compiled graph,
model/run references and raw closure, with no CV parent or score.
`build_refit_artifact_facets(child, parent_outcome, source_package)` validates the
actual child plus its parent recipe/package/outcome. It retains distinct source
and target evidence and links the child `refit_of` the source evidence. Source
scores remain source scores. No evaluation is inherited by the refit.

Current RolePipeline declarative validation is a shared internal helper reused
by the old assembler and the new byte proof path. Existing multimodal and
classifier owner validators are reused. The facet layer adds no Methods binary
decoder. `inspect_facet_model_states` explicitly returns unverified state facts.
With `methods-optimizer`, `inspect_facet_model_states_with_methods` requires an
actual configured `MethodsRuntime`, explicitly calls
`n4m::configure_library(runtime.library_path())`, then invokes the unchanged
owner `inspect_n4mm` adapter for supported N4MM artifacts and refuses descriptor
mismatches. n4m 0.4.1 uses an immutable process-wide OnceLock: same canonical
path is an idempotent check; a different path is refused and cannot switch the
library. MethodsRuntime stores a canonical path and ABI, not a captured Context
or per-instance inspector. Inspection and replay ABI acceptance remain separate.
These are facts reported by the actual owner inspector, not code authentication. Other state
families stay opaque/unverified. Native inspection never hydrates.

`check_facet_hydration_requirements` only checks prerequisites: declared
controller, serializer and versioned plugin capabilities, explicit caller trust
for host serializers (including executable backends mislabeled native), or an
explicit native runtime capability. Success does not authorize loading. The
module contains no loader callback and never invokes `load_with`.
`require_facet_operation` admits `inspect` on validated content and
`score-inspect` on a validated ScoreSet, checking all supplied root/input IDs
and proofs and refusing known link mismatches. Predict/refit/compile and unknown
operations have no execution admission in this slice and are refused. Neither
missing evidence nor an archive-declared boolean bypasses this boundary.

## Exact owner export and validation

The package-local fixtures under `crates/dag-ml-core/tests/fixtures/artifact_facets`
include host/mock control fixtures plus genuine native `native_role/` owner
exports. The latter were produced by actual native RolePipeline fitting and a
fresh target-bound child refit, with exact raw N4ME closure and generator
provenance. Current-source default tests validate these bytes and execute the
training/refit facet builders, resolver and composite checks. Neither family
qualifies IO logical identity or evaluation population parity. The optional
native test additionally executes current `execute_training` and invokes
`inspect_facet_model_states_with_methods` on its validated N4MM3 package proof,
then tests an owner-structurally-valid descriptor mismatch and opaque RolePipeline
status. Runtime path refusals are boundary tests, not successful executions.
The example accepts actual direct producer outputs and creates a new immutable
capture directory containing `facets.json`, `links.json`, `inventory.json` and
exact original owner members. `links.json` is explicitly tagged
`diagnostic_only`: it is a non-authoritative report, never persisted validation
or admission. `FacetLinkVerdict`, `FacetLinkValidation` and their status cannot be
deserialized from JSON; live proofs must be recreated from bytes. Trusted dataset
ports can still construct live verdicts as explicit caller-selected authorities,
which do not authenticate producer identity.

The producer command is:

```sh
CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=/tmp/dagml-facets-target \
  cargo +1.88.0 run --locked -p dag-ml-core --example artifact_facets -- \
  /tmp/current-outcome.json /tmp/current-package.json /tmp/new-owner-facets
```

No Core dependency or unreleased sibling path is required. Core tests consume
committed exact bytes; integrated IO/native composition and fresh-process
prediction are separate external qualification. Default tests use only
package-local bytes and current owner builders. The schema checks declaration
shape; Rust supplies cross-object and semantic checks. The schema does not
assert that an opaque payload or scientific link has been validated.

## Typed public refusals

New owner functions return `FacetResult<T>` with `FacetRefusal` variants:
`UnsupportedOperation`, `UnresolvedOwnerContent`, `Mismatch`, `MissingTrust`,
`MissingCapability`, `RuntimeUnavailable`, `UnsupportedPayload`,
`UnverifiedEvidence`, `InvalidPayload`, `InvalidStructure`, `Json`, and
`OwnerValidation`. Existing owner errors are retained as typed sources inside
`OwnerValidation`; JSON errors retain their source. Callers must match variants,
not parse RuntimeValidation message strings. The historical extracted
RolePipeline helper retains its existing DagMlError result for the old wrapper.
`configure_facet_methods_runtime` and native inspection map actual owner library
configuration failures to `RuntimeUnavailable`. Caller-supplied capability
strings remain declared prerequisites and never grant execution admission.

## Exact proven physical closure and cross-run refusals

`validate_facet_members(artifact, proof, members)` is the public reader check
shared by `ArtifactFacetExport::validate`. Descriptor/content/identity/kind must
match the live owner proof before closure is checked. Every proven embedded
facet requires exact original root bytes. Only Model and FullRefit facets own
external raw-state closure; their existing `validate_facet_state_members` API
keeps exact content-addressed members, per-facet namespace and same-SHA dedupe.
Every other proven embedded facet requires `member_paths == [root_member]`,
even if its payload contains a raw map. Neither an Attachment nor a Run thereby
acquires raw-state ownership. Extra unvalidated support files are a typed
`Mismatch`. The state-reader API applies the same root-only rule to those kinds
and returns an empty state inventory when valid.

Unknown embedded content without owner proof may still pass generic physical
storage checks; it remains scientifically unverified and cannot supply owner
operation admission. Embedded Model/FullRefit facets still require a proof.

A live InitialRefit or ChildRefit run identifies its exact validated package.
Known DAG source proofs with a different identity are producer mismatches,
including parent scores/model/outcome/cache linked to a child run. InitialRefit
or ChildRefit content linked to a parent Outcome run or a CV execution Bundle
run also mismatches: a refit is not produced by that CV bundle. A refit never
inherits source evaluation through such a link. Matching parent score/bundle
closure remains unverified because the bundle carries no owner run label;
other genuinely undecidable Bundle-run claims also remain unverified. Valid
existing producer reports are unchanged.

Candidate bytes disagreeing with an optional reference `root_sha256` are a
`FacetRefusal::Mismatch`, just like disagreement with `sha256.raw.v1`. Declared
content version versus actual owner payload version disagreement is classified
as `FacetRefusal::InvalidStructure`: it is structural version selection, not a
new refusal category or a rule that rejects state families by age.

## Explicit owner actions (B1/B2)

The closed C wire vocabulary and all five projection profiles are unchanged.
`require_facet_operation` remains inspection-only. Capabilities, stored link
reports and archive booleans never authorize scientific execution.

`facet_calibration_view(proof)` borrows the binding/targets, calibration hash,
bundle reference, retained replay hash and pre-conformal source anchor from a
currently validated Outcome or PredictorPackage. These fields are a read-only
projection of existing schema2 state, not another scientific payload or proof
minted from diagnostic JSON. Owner validators reconstruct/check the pre-conformal
source; the post-calibration outcome hash is different and is never substituted
for that anchor.

`calibrate_training_artifact_facets(graph, context, members, outcome_id, model_id,
request)` first validates the complete supplied physical/link closure. The
selected result/evidence must match the selected model. `FacetCalibrationRequest`
contains actual replay, relations, identity-aligned truth, binding and conformal
policies, but no caller-built context. The action clones the pre-calibration
outcome and calls `calibrate_attached_training_replay_with_derived_context`, then
exports a matching package and complete fresh facet set. Recalibrating an already
calibrated outcome is explicitly unsupported by this port. Current owner
truth-ID alignment and physical/origin leakage checks remain authoritative.

Calibration neither retrains nor adds independent evaluation. Actual fitted
state bytes, artifact modes and source scores are retained. A source score that
is byte-identical remains valid; unrelated scores or pre-calibration evidence
mixed with a calibrated package refuse. No calibration kind, relation, standalone
member or third calibration-state copy is added. Unknown original container
attachments are Core's explicit preservation/construction decision; this DAG
operation returns only its complete newly derived owner set.

`admit_facet_predict(graph, context, members, root_id, request, cohort)` is ungated
and works with default features/WASM. It validates C closure for every proven
embedded facet, exact selected Model/FullRefit root and raw states, semantic
links, concrete request keys/output bindings and current owner preflight. The
root must be embedded with complete supplied physical closure. Unknown opaque
storage cannot become an operation root. `FacetPredictRequest::Replay` carries
an actual TrainingReplayRequest for a predictor or child-refit PREDICT;
`InitialRefit` carries actual initial-refit output IDs and uses that owner's
distinct preflight. This step has no runtime, controller, hydration or numerical
callbacks. Dataset/evaluates/residual links retain C's separate unverified status
when their complete evidence is absent.

`AdmittedFacetReplay<C>` has private fields and no Deserialize/Clone/token
constructor. It moves the concrete frozen cohort into the request and retains
owned root/state and envelope snapshots for this action only. The generic
`FacetReplayCohort` is an explicitly trusted in-process Rust adapter interface,
not authentication or a defense against malicious Rust/interior-mutability lies.
Preflight checks metadata compatibility; it cannot attest actual numeric values
against an arbitrary claimed fingerprint.

With `methods-optimizer`, the trusted IO/DMD snapshot adapter implements
`MethodsFacetCohortOwner::into_bindings(self)`. Each `MethodsFacetBinding` pairs
an actual dataset and actual envelope derived from ONE immutable owner snapshot.
The adapter must validate IO identity and numerical projection, source/feature
order, target order, masks and binding coordinates. Public adapter-input fields
are not admission. `freeze_methods_facet_cohort` consumes this owner, validates
its structural projection/cohort agreement, and privately retains owned numeric
vectors. There are no setters, serde, Clone or loose-map constructor on
`FrozenMethodsFacetCohort`. Generic DAG gains no IO dependency or IO decoder.
IO raw SHA, IO logical identity and TrainingDataIdentity are not equated.

`execute_admitted_methods_facet_replay(admitted, runtime, outcome_id, run_id)`
consumes only `AdmittedFacetReplay<FrozenMethodsFacetCohort>`; no later buffers,
envelopes, provider or controller registry can replace the frozen values. It
checks the actual immutable Methods loader path and compiled native controller
set before dispatch. Host-sidecar mode refuses with `MissingTrust`; this native
port accepts no trust-string override. Missing/unsupported owners stay typed
refusals. Actual existing replay/refit scheduler/provider loops reindex retained
rows by stable sample IDs and perform all numerical work. Predictor and child
refit replay use their current typed Methods APIs; initial-refit PREDICT uses its
own current owner executor. Merely compiling this dispatch does not qualify every
state/controller combination. Active unsupported/unqualified families and their
working client routes are retained.

This port does not authorize FIT, refit training, EXPLAIN, compilation, browser
controller dispatch or any host facade/cutover. Existing WASM single-node
restrictions and controllers are unchanged. Default/WASM builds contain no native
dynamic loader code. Actual IO/DMD same-snapshot qualification is separate from
metadata-only/synthetic test adapters and must project both identity and numerical
inputs from the same actual IO snapshot. Retained native fixtures prove their
recorded producer run, not that new adapter or installed platform parity. Owner
consistency proofs do not authenticate producer identity.

For predictor/child-refit requests, admission validates target coverage and relative
order for explicitly selected current OutputBindings grouped by their actual owner
input requirement keys. It reuses the existing predictor-closure helper; it does not
assume one dataset per run or infer an evaluation population. A selected output
without concrete V2 cohort evidence refuses as `UnverifiedEvidence`. The
one-target committed calibration fixture covers target-name/shape contradictions;
it does not claim a multi-target permutation qualification.

### Shared replay target-claim authority

For each actual input requirement key in the selected outputs' predictor closures,
explicit V2 `PredictCohort.target_names` must cover exactly the union of those
selected `OutputBinding.target_names`. Each binding's ordered list must appear as
an ordered subsequence of the cohort. There is no imposed order between independent
outputs: single-target `[protein]` and `[fat]` outputs permit either cohort order.
Conflicting orders on a shared key require separate requests. Unselected bindings
do not contribute coverage, and distinct keys are checked independently. With one
binding this reduces exactly to target-name equality, including order.
This is one shared owner rule in
`replay.rs`, used by both public detached preflights and the attached-outcome,
loaded-predictor and child-V3 execution entrypoints before provider/controller
callbacks or state hydration. Canonical facet admission reuses that same rule;
its stricter required-V2 presence policy remains separate. A known contradiction
maps directly to typed facet `Mismatch`; missing required canonical evidence maps
to `UnverifiedEvidence`, without inspecting exception text.

PREDICT and EXPLAIN requests both retain explicit saved output-binding IDs.
EXPLAIN collection itself is separate: `bind_attached_replay_explanations` does
not filter explanation producers by that list. The rule checks the requested
saved-binding claims and their actual input closures; it imposes no new target
constraint on unselected bindings or unrelated explanation blocks. Existing V1 envelopes with
no explicit PredictCohort assert no target-name vector and retain their existing
owner behavior. No new legacy format or compatibility fixture is introduced.

CLI detached replay calls the shared preflight/execution; general Python and
Rust/WASM loaded replay call the shared loaded executor; C-ABI attached replay
calls the shared attached executor. These routes inherit the fix without copied
host comparators. Binding regressions exercise public C/Python calls and the
Rust-side WASM provider/executor delegation, with measured callback refusal and
positive controls. Compiled binding checks are not installed wheel/browser
qualification.

InitialFullRefitOutput stores only output ID, node and port, with no saved target
names. Initial-refit and terminal-selector paths therefore keep their distinct
current contracts, including emitted-name versus cohort checks; no missing saved
target metadata is invented. Existing initial/terminal controls remain in place.

### Structural entrypoint roles and calibration preflight

`resolve_facet_entrypoint` returns structural roles only; they are not admission.
The caller composes `FacetDatasetContract` from IO-owned compiled vocabulary.
DAG does not validate dataset bytes or establish numerical authority here.

| Operation | Root | Inputs and entrypoint requirement |
| --- | --- | --- |
| `predict` | Embedded DAG Model or FullRefit | Exactly one embedded IO-contract Dataset; requirement null |
| `predict-external` | Embedded DAG Model or FullRefit | No inputs; optional embedded IO-contract Dataset schema anchor |
| `calibrate-external` | Embedded DAG Model | No inputs; required DAG Result or TrainingEvidence |

Core and IO must recompute identity, validate its profile and match the descriptor
and normalized schema and version from the same bytes. An anchor requires exact
owner public-source-schema equality. Without an anchor, the owner's own DMD
requirements still apply, but there is no named-axis or unit guarantee and no
positional fallback. Root descriptor requirements are independent of entrypoint
requirements. A reference root or anchor is unresolved.

`preflight_facet_calibration` checks the same static closure and output-binding
existence as the action before replay; it returns no admission token. The action
repeats these checks and retains the later replay validation.

The combined manifest fixture is the exact Core qualification producer manifest
`nirs4all-core/bindings/rust/nirs4all/tests/parity/fixtures/n4a_v1/combined/manifest.json`,
SHA-256 `4b2cd88270436433955fc9ba8f4851934c4079f534452cdc0c195d6aa5bafa4d`.


## Finite JS RolePipeline v2 semantic readability (2026-11-08)

The existing facet reader also recognizes one existing JS controller profile:
`methods_role_pipeline`, RAW, `dagml.methods.wasm.regression@1.1.0`, with both
record and artifact owned by `controller:methods.wasm.regression` and the same
controller/version in the effective node plan. The strict private wrapper must
use `dagml.methods.regression.v2`, `controller_version: "1.1.0"` and
`recipe_profile: "pipeline_steps.v1"`; unknown fields and other versions refuse.
The existing 1.0.0/v1 profiles retain their current validation and producer rules.

This v2 profile requires `N4mRolePipeline` and a nonempty, explicit
`node.params.pipeline_steps_v1`. Each step declares exactly one nonempty `class`
or `methodId`, with optional object parameters and no nested reserved recipe key.
If `operator.steps` is present, it must equal that base recipe. Only the reserved
node parameter is removed before the remaining selected parameters overlay the
last step. The complete saved recipe and effective parameter fingerprint must
then agree. A 1.1.0 wrapper produced from `operator.steps` alone or `n4m:<id>` is
unsupported here; the wrapper label does not establish recipe provenance.

The raw wrapper SHA-256, size, URI, strict TCV1, node and parameter identity,
unique feature/target names and absence of foreign native descriptors remain
mandatory. The byte validator retains the original package bytes; the existing
builder may reserialize the owner package while preserving its identity and the
exact raw wrapper and state bytes. The pinned public synthetic run02 fixture
retains base PLS component count 2,
selected count 6, untouched SNV/SG declarations, 28 ordered training IDs and the
genuine distinct source-table S and canonical-relation C fingerprints.

There must be exactly one nonempty N4ME-marked buffer per declared step, with the
existing i32 step-count bound. Filter recipes with fewer states remain
unsupported in this profile. Buffers stay opaque: this check establishes no
native state validity, width, role association, retention, replay, executability
or portability. `inspect_facet_model_states` continues to return unverified
state claims. The shared native caller gains only this declarative coverage;
its Package3, controller and runtime gates are unchanged. No wire, public API,
ABI, schema, version or numerical change is introduced.

Qualification is deferred to the three `js_v2`/`genuine_js_v2` integration cases
in `tests/artifact_facets.rs` and the existing v1/native regressions. Source
preparation alone is not a successful gate. Native owner facet acceptance is
required before the Core ignored producer, the measured old3377 refusal and any
conditional Core WASM relink. The old reader may refuse relations or the codec
first; that baseline does not isolate S/C from the changed reader coverage.
