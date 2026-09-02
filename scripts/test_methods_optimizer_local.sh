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
# The Archive V2 cross-repo test has a dev-only Core dependency which in turn
# pulls the published dag-ml-core. Select this checkout by manifest path so
# Cargo never ambiguously targets the registry package.
cargo clippy --manifest-path "$core_manifest" --features methods-optimizer --all-targets -- -D warnings
cargo test --manifest-path "$core_manifest" --features methods-optimizer "$@"
