# Initial full-refit package V1

`InitialFullRefitPackage` records one concrete REFIT over the entire supplied
training universe when a campaign has no splitter. It is a separate contract
from `PortablePredictorPackage` (CV training) and `PortableRefitPackageV3`
(a refit bound to a prior package and a new target request). No CV fold,
selection, OOF cache, parent bundle, or score is asserted by this package.

The native executor requires one plan variant, an explicit ordered list of
training sample IDs, complete sample relations, and feature and target content
fingerprints for every data binding. It checks the supplied order against the
provider's training order, and the sample set against the relations. The
package closes the effective plan, the content identities, output node/port
bindings, the training schema/plan envelope, execution seed/resources/scheduler,
and REFIT artifact inventory under TCV1 fingerprints. The envelope retains the
training relations while a new, separate V2 prediction cohort is attached for
replay. A raw native
artifact includes its bytes in `raw_artifact_payloads`; a host artifact is
marked `HostSidecar` and needs its separately persisted sidecar for replay.
The package itself does not claim to contain host model bytes.

In the CLI, `run-process-dsl-refit-phase --package-output PATH` executes and
writes the package; `--package-id` selects its ID. The outcome JSON includes
`initial_full_refit_package` and may retain actual final/test reports, but no
CV or selection score is invented. The
`validate-initial-full-refit-package PATH` command checks the closed package.
In PyO3, `execute_phase_in_process(..., phase="REFIT",
training_sample_ids=..., package_id=...)` returns the same package in its JSON
outcome; `validate_initial_full_refit_package_json` validates it. The public
Python wrapper exposes `InitialFullRefitPackage` for the same validation.
PyO3 `execute_phase_in_process(..., artifact_callback=...)` can also export
raw operator bytes into the package. The callback receives
`{"operation":"export","artifact_id":...}` and returns a nonempty byte list.
`replay_initial_full_refit_in_process(..., artifact_callback=...)` sends the
existing `hydrate` and `release` requests to materialize those bytes in a
fresh Python process; an all-raw package needs no host-sidecar handles.

`run-process-initial-full-refit-predict` and
`replay_initial_full_refit_in_process` take the closed package, a separate V2
PREDICT cohort, exact host-sidecar artifact handles and explicit output IDs. Raw
native artifacts are hydrated from the package without external handles. The
native scheduler validates the requested terminal ports and provider cohort,
materializes the REFIT artifacts, and emits a fingerprinted replay outcome.
The host must resolve its own sidecar bytes into those invocation-local handles;
the package cannot claim to contain them. A missing artifact or a different
cohort fails before operator execution.
The CLI process-controller path can capture host sidecars. A one-shot operator
process exits before the core can request Raw bytes, so it cannot publish a
portable Raw artifact. A persistent JSONL adapter that declares
`control_frames_v1` and `portable_artifact_bridge_v1` can instead receive
`portable_artifact` frames for Raw export, hydration, and release. The CLI
routes export to the worker that produced the REFIT artifact and hydration to
the worker executing PREDICT. As with Rust, C, WASM, and PyO3 bindings, the
controller must implement these operations for its own artifact format; the
transport alone does not serialize arbitrary fitted models.

## Raw-artifact callback protocol for C and WASM hosts

The existing C `DagMlControllerVTable.invoke` and WASM `js_invoke` callbacks
also carry `PortableArtifactBridgeTask` JSON when a controller emits an artifact
with `backend: "raw"`. This adds no fields to the C vtable or request structs.
Each request has `operation` and `schema_version: 1`; its response must be a
`PortableArtifactBridgeResult` JSON string with the matching operation and
version:

| Request operation | Request fields | Response operation | Response fields |
| --- | --- | --- | --- |
| `export_artifact_payload` | `artifact_id` | `exported_artifact_payload` | `payload`: nonempty byte array |
| `hydrate_artifact_payload` | `request`: artifact materialization request, `payload`: byte array | `hydrated_artifact_payload` | `handle`: nonzero handle owned by the controller |
| `release_hydrated_artifact_payload` | `handle` | `released_hydrated_artifact_payload` | none |

The core exports every raw REFIT artifact into the signed package. On PREDICT,
it hydrates payloads with the owning controller, uses the returned handles for
that invocation, and releases them even when execution fails. The host callback
must serialize raw bytes as JSON integers in `0..255`. A fresh host can replay
the package with an empty sidecar-handle map when all artifacts are raw. The
package validates each raw payload against its artifact `size_bytes` and
SHA-256 `content_fingerprint`, even if a caller recomputes the outer package
fingerprint. The ordinary `NodeTask`/`NodeResult` callback protocol remains unchanged for
non-raw artifacts. Controllers that do not support raw payload operations
should return a validation error; the package capture or replay then fails.
