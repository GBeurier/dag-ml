#!/usr/bin/env bash
# Run native Methods integration without putting a local runtime or linker
# dependency in the publishable dag-ml-core manifest. The trap restores both
# files byte-for-byte so this is safe in a developer checkout with unrelated
# changes.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
core_manifest="$repo_root/crates/dag-ml-core/Cargo.toml"
local_manifest="$repo_root/crates/dag-ml-core/Cargo.toml.methods-local"
library_path=${N4M_LIBRARY_PATH:-}
probe_only=false

if [[ "${1:-}" == "--probe" ]]; then
  probe_only=true
  shift
fi

if [[ ! -f "$local_manifest" ]]; then
  echo "missing Methods-local manifest overlay: $local_manifest" >&2
  exit 2
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

manifest_backup=$(mktemp)
lock_backup=$(mktemp)
cp "$core_manifest" "$manifest_backup"
cp "$repo_root/Cargo.lock" "$lock_backup"
restore() {
  cp "$manifest_backup" "$core_manifest"
  cp "$lock_backup" "$repo_root/Cargo.lock"
  rm -f "$manifest_backup" "$lock_backup"
}
trap restore EXIT

cp "$local_manifest" "$core_manifest"
cd "$repo_root"
# The Archive V2 cross-repo test has a dev-only Core dependency which in turn
# pulls the published dag-ml-core. Select this checkout by manifest path so
# Cargo never ambiguously targets the registry package.
cargo clippy --manifest-path "$core_manifest" --features methods-optimizer-local --all-targets -- -D warnings
cargo test --manifest-path "$core_manifest" --features methods-optimizer-local "$@"
