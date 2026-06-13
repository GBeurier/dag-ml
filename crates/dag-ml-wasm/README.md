# dag-ml WASM bindings

Browser-friendly bindings for DAG-ML JSON contracts.

The WASM package exposes validation, DSL compilation and execution-plan
construction over UTF-8 JSON strings. It intentionally excludes host controller
execution, artifacts and data-buffer ownership.

`contract_manifest_json()` returns a stable JSON manifest with the package
version, supported contract ids, exported Python/WASM function names and shared
fixture digests. Browser integrations should check it before accepting cached
pipelines or persisted `nirs4all-lite` workspaces.

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
  contract_manifest_json,
  validate_pipeline_dsl_json,
  compile_pipeline_dsl_artifact_json,
} from "./pkg/dag_ml_wasm.js";

await init();
const manifest = JSON.parse(contract_manifest_json());
validate_pipeline_dsl_json(JSON.stringify(dsl));
const artifact = JSON.parse(compile_pipeline_dsl_artifact_json(JSON.stringify(dsl)));
```

Rust-side validation failures are returned as JSON strings with the ADR-11
descriptor fields `category`, `code`, `severity`, `message`,
`remediation_hint` and `context`.
