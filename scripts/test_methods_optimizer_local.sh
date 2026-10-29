#!/usr/bin/env bash
# Run native Methods integration without a local runtime, linker dependency,
# sibling manifest, or sibling source dependency. The helper enables the
# published Methods dependency plus a compiler-only native-test selector.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
core_manifest="$repo_root/crates/dag-ml-core/Cargo.toml"
library_path=${N4M_LIBRARY_PATH:-}
probe_only=false

if [[ "${1:-}" == "--probe" ]]; then
  probe_only=true
  shift
fi

if [[ -z "$library_path" ]]; then
  echo "N4M_LIBRARY_PATH must explicitly name the libn4m shared-library file" >&2
  exit 2
fi
if [[ "$library_path" != /* ]]; then
  echo "N4M_LIBRARY_PATH must be an absolute path" >&2
  exit 2
fi
if [[ ! -f "$library_path" ]]; then
  echo "N4M_LIBRARY_PATH must name a regular libn4m file (got $library_path)" >&2
  exit 2
fi
if [[ "$probe_only" == true ]]; then
  printf 'N4M_LIBRARY_PATH=%s\n' "$library_path"
  exit 0
fi

export N4M_LIBRARY_PATH="$library_path"

# Keep the native runtime tests out of ordinary/all-feature package builds
# without copying a second manifest. The published feature selects only n4m;
# this compiler cfg selects the runtime-dependent tests in this checkout.
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--cfg feature=\"methods-optimizer-local\""

cd "$repo_root"
# Candidate dependencies may require an explicit, audited owner-source config
# before publication. Never silently rewrite package identities or manifests.
cargo_args=()
if [[ -n "${DAG_ML_LOCAL_CARGO_CONFIG:-}" ]]; then
  cargo_args+=(--config "$DAG_ML_LOCAL_CARGO_CONFIG")
fi
# Archive readers are separate bounded processes. Format 3 additionally needs
# DAG_ML_CURRENT_CORE_ORACLE built as tools/current-core-oracle/README.md says;
# the exact historical 0.4.3 process is never substituted for that reader.
cargo "${cargo_args[@]}" clippy --manifest-path "$core_manifest" --features methods-optimizer --all-targets -- -D warnings
cargo "${cargo_args[@]}" test --manifest-path "$core_manifest" --features methods-optimizer "$@"
