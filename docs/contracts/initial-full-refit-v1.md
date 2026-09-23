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
bindings, execution seed/resources/scheduler, and REFIT artifact inventory under TCV1 fingerprints. A raw native
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

`run-process-initial-full-refit-predict` and
`replay_initial_full_refit_in_process` take the closed package, a separate V2
PREDICT cohort, exact host-sidecar artifact handles and explicit output IDs. Raw
native artifacts are hydrated from the package without external handles. The
native scheduler validates the requested terminal ports and provider cohort,
materializes the REFIT artifacts, and emits a fingerprinted replay outcome.
The host must resolve its own sidecar bytes into those invocation-local handles;
the package cannot claim to contain them. A missing artifact or a different
cohort fails before operator execution.
