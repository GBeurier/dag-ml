# Archive V3 native full-refit contract

Archive V3 is the closed storage family for the target-bound full-refit child
defined by ADR-25. It does not extend Archive V2: V1 and V2 stay immutable,
dual-readable predictor families and reject retraining.

The Archive V3 writer is Core; DAG-ML owns the semantic closure and emits its
exact bytes before Core writes the bounded ZIP. Core stores and validates the
manifest, inventory and raw N4MM bytes. It does not parse a DAG, execute a
model, or synthesize an artifact. DAG-ML is the only package/refit/replay
reader.

## Exact member closure

The following members are mandatory and inventory-closed:

1. `dagml/portable_refit_package.json` — `PortableRefitPackage` V3;
2. `dagml/graph.json` — GraphSpec V1 used by the V3 effective plan;
3. `dagml/portable_refit_execution_bundle.json` — refit-only bundle V3;
4. `dagml/portable_refit_outcome.json` — target-bound outcome V3; and
5. one or more raw, plugin-free N4MM models at safe `methods/*.n4mm` paths.

The package, outcome and refit bundle are distinct members so the archive
container can bind each child fingerprint independently. They are not V2
`ExecutionBundle` or `TrainingOutcome` members, and V3 never carries source
CV/OOF/selection reports, prediction cache payloads, scores, conformal state,
host sidecars, process-local handles or optimizer checkpoints.

Every N4MM reference byte-equals its matching V3 detached raw artifact payload.
New writers also emit a capability-derived `abi_min_minor`: PLS N4MM is 2.0+
and imported-linear/Ridge N4MM is 2.3+. Historical absence is accepted only for
PLS; it is not a permissive default for Ridge.
The semantic profile is `n4mm_raw_sha256`: the semantic fingerprint equals the
raw SHA-256. No JSON-canonical fingerprint is fabricated for native bytes.

`archive_workspace_manifest.v3.schema.json` freezes the manifest identity.
Archive V3 is accepted only at exact version/profile/writer identity; future
versions, V1/V2 package families, host artifacts, and all nullable historical
payload slots other than explicit null are refused before payload handling.
