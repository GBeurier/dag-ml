#!/usr/bin/env bash
# Run the unpublished n4m integration without putting its pinned Git dependency
# in the publishable dag-ml-core manifest. The trap restores both files
# byte-for-byte so this is safe in a developer checkout with unrelated changes.
set -euo pipefail

readonly N4M_METHODS_PIN="0ef355e6f74573ed07a6920bdeed1a052a6e8312"
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
core_manifest="$repo_root/crates/dag-ml-core/Cargo.toml"
local_manifest="$repo_root/crates/dag-ml-core/Cargo.toml.methods-local"
methods_root=${N4M_METHODS_REPO:-}
methods_sha=${N4M_BINDING_SHA:-$N4M_METHODS_PIN}
probe_only=false

if [[ "${1:-}" == "--probe" ]]; then
  probe_only=true
  shift
fi

if [[ ! -f "$local_manifest" ]]; then
  echo "missing Methods-local manifest overlay: $local_manifest" >&2
  exit 2
fi
if [[ -z "$methods_root" ]]; then
  echo "N4M_METHODS_REPO must explicitly point to the pinned nirs4all-methods checkout" >&2
  exit 2
fi
if [[ ! -d "$methods_root" ]]; then
  echo "N4M_METHODS_REPO must point to a nirs4all-methods checkout (got $methods_root)" >&2
  exit 2
fi
methods_root=$(cd "$methods_root" && pwd -P)
if [[ ! -f "$methods_root/bindings/rust/n4m/Cargo.toml" ]]; then
  echo "N4M_METHODS_REPO must point to a nirs4all-methods checkout (got $methods_root)" >&2
  exit 2
fi
if [[ ! "$methods_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "N4M_BINDING_SHA must be a full 40-character commit SHA" >&2
  exit 2
fi
methods_sha=${methods_sha,,}
if [[ "$methods_sha" != "$N4M_METHODS_PIN" ]]; then
  echo "N4M_BINDING_SHA must match the reviewed Methods pin $N4M_METHODS_PIN" >&2
  exit 2
fi
if ! actual_binding_sha=$(git -C "$methods_root" rev-parse HEAD 2>/dev/null); then
  echo "N4M_METHODS_REPO must be a Git checkout at $N4M_METHODS_PIN" >&2
  exit 2
fi
if [[ "$actual_binding_sha" != "$methods_sha" ]]; then
  echo "Methods binding SHA mismatch: expected $methods_sha, got $actual_binding_sha" >&2
  exit 2
fi
if [[ -n "$(git -C "$methods_root" status --porcelain=v1 --untracked-files=all -- \
  Cargo.toml Cargo.lock CMakeLists.txt Makefile cmake cpp bindings/rust/n4m)" ]]; then
  echo "pinned Methods build or binding inputs are dirty in $methods_root" >&2
  exit 2
fi
if [[ "$probe_only" == true ]]; then
  printf 'N4M_METHODS_REPO=%s\n' "$methods_root"
  printf 'N4M_BINDING_SHA=%s\n' "$methods_sha"
  exit 0
fi
if [[ -z "${N4M_LIB_DIR:-}" ]]; then
  echo "N4M_LIB_DIR must point to the directory containing libn4m.so (or platform equivalent)" >&2
  exit 2
fi

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
