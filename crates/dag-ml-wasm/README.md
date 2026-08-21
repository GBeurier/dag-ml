# dag-ml WASM bindings

Browser-friendly bindings for DAG-ML JSON contracts.

The WASM package exposes validation, DSL compilation, execution-plan
construction and synchronous host-controller execution over UTF-8 JSON strings.
Artifacts and data-buffer ownership remain outside the binding.

`contract_manifest_json()` returns a stable JSON manifest with the package
version, supported contract ids, exported Python/WASM function names and shared
fixture digests. Browser integrations should check it before accepting cached
pipelines or persisted `nirs4all-core` workspaces.

## Build

```bash
cargo test -p dag-ml-wasm
node_out_dir="$PWD/target/wasm/dag-ml-wasm"
wasm-pack build crates/dag-ml-wasm --target nodejs --out-dir "$node_out_dir" --release
node scripts/smoke_wasm_bindings.cjs "$node_out_dir"
web_out_dir="$PWD/target/wasm-web/dag-ml-wasm"
wasm-pack build crates/dag-ml-wasm --target web --out-dir "$web_out_dir" --release
node scripts/smoke_wasm_web_bindings.mjs "$web_out_dir"
```

## JavaScript Surface

```js
import init, {
  LocalImplementationRegistry,
  contract_manifest_json,
  loss_execution_attestation_json,
  validate_pipeline_dsl_json,
  compile_pipeline_dsl_artifact_json,
} from "./pkg/dag_ml_wasm.js";

await init();
const manifest = JSON.parse(contract_manifest_json());
validate_pipeline_dsl_json(JSON.stringify(dsl));
const artifact = JSON.parse(compile_pipeline_dsl_artifact_json(JSON.stringify(dsl)));
```

Local JavaScript losses and metrics are retained by a WASM registry and never
serialized into DAG-ML contracts:

```js
const implementations = new LocalImplementationRegistry();
implementations.register_loss(JSON.stringify(lossReference), weightedLoss);

const loss = implementations.resolve_training_loss(
  JSON.stringify(trainingLossRole),
  "FIT_CV",
);
const value = loss(target, prediction);
const attestation = JSON.parse(
  loss_execution_attestation_json(JSON.stringify(trainingLossRole), "FIT_CV"),
);
```

Controllers should bind from the native `NodeTask` rather than resolving a
role and constructing lineage independently:

```js
const binding = implementations.bind_training_loss(
  JSON.stringify(nodeTask),
  0,
);
const loss = binding.invoke;
const value = loss(target, prediction);
const requiredAttestation = JSON.parse(binding.required_attestation_json);
binding.free();
```

The role index is zero-based. Binding validates that the task's ordered loss
requirements exactly match its active roles. A controller may copy the returned
attestation into `NodeResult.lineage.loss_attestations` only after `loss(...)`
returns successfully.

For scheduler execution, lower the roles into the native plan and execute that
plan rather than rebuilding a loss-free campaign plan:

```js
const planJson = build_execution_plan_with_training_losses_json(
  planId,
  JSON.stringify(graph),
  JSON.stringify(campaign),
  JSON.stringify(controllerManifests),
  JSON.stringify(trainingLossRoles),
);

const resultsJson = execute_execution_plan_phase_json(
  planJson,
  JSON.stringify(controllerManifests),
  runId,
  rootSeed,
  "FIT_CV",
  (controllerId, taskJson, exactSeed) =>
    controller.invoke(controllerId, taskJson, exactSeed),
);
```

The native lowerer replaces all plan loss roles, groups them by node and sorts
them canonically before validating controller capabilities. At execution, every
manifest embedded in the plan must exactly match the independently supplied
trusted controller registry before any callback runs. The scheduler then checks
that each callback result contains exactly the task's required loss attestations
in the same order.

The callback's `exactSeed` argument is a decimal string (or `null`), avoiding
precision loss for native `u64` seeds beyond JavaScript's safe-integer range.
The callback may set `NodeResult.lineage.seed` to `null`; the WASM bridge then
injects the authoritative native seed before scheduler validation. The returned
`resultsJson` still contains native numeric `u64` values; use a lossless JSON
integer parser, or preserve the raw JSON, when inspecting lineage seeds exactly.

JavaScript-local descriptors use `binding:javascript` and a `host_local` or
`portable_registered` lifecycle. A Web Worker must populate its own registry;
functions are not cloned, posted, or embedded in replay artifacts. Resolution
is rejected when the exact descriptor or phase does not match.

Rust-side validation failures are returned as JSON strings with the ADR-11
descriptor fields `category`, `code`, `severity`, `message`,
`remediation_hint` and `context`.
