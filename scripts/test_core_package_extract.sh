#!/usr/bin/env bash
# Verify that dag-ml-core's published crate tests without its source workspace.
set -euo pipefail

scratch_dir=$(mktemp -d)
trap 'rm -rf "$scratch_dir"' EXIT
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
package_dir="$scratch_dir/package-target/package"
extract_target_dir="$scratch_dir/extract-target"

# The manifest declares the workspace-only fixture cfg; leave it disabled so
# the package cannot resolve repository docs, generated examples, or W1 data.
env -u RUSTFLAGS CARGO_TARGET_DIR="$scratch_dir/package-target" \
  cargo package --manifest-path "$repo_root/crates/dag-ml-core/Cargo.toml" \
  --allow-dirty --no-verify

mapfile -t archives < <(find "$package_dir" -maxdepth 1 -type f -name 'dag-ml-core-*.crate' -print)
if [ "${#archives[@]}" -ne 1 ]; then
  echo "expected exactly one dag-ml-core archive, found ${#archives[@]}" >&2
  exit 1
fi
crate_archive=${archives[0]}
archive_contents="$scratch_dir/archive-contents.txt"
tar -tzf "$crate_archive" > "$archive_contents"
if rg -q '/(examples|docs)/|negative_cases\.v1\.json' "$archive_contents"; then
  echo 'package unexpectedly contains workspace fixtures or the W1 corpus' >&2
  exit 1
fi
if rg '(^|/)[^/]+\.v[0-9]+\.json$' "$archive_contents" \
  | rg -vq '/tests/fixtures/conformal_w0_golden\.v1\.json$'; then
  echo 'package unexpectedly contains an unapproved protocol-versioned fixture' >&2
  exit 1
fi
if ! rg -q '(^|/)README\.md$' "$archive_contents"; then
  echo 'package is missing its package-local README' >&2
  exit 1
fi
if ! rg -q '(^|/)LICENSE$' "$archive_contents"; then
  echo 'package is missing its package-local LICENSE' >&2
  exit 1
fi
tar -xzf "$crate_archive" -C "$scratch_dir"
extracted_manifest=$(find "$scratch_dir" -mindepth 2 -maxdepth 2 -name Cargo.toml -print -quit)
test -n "$extracted_manifest"

# A library package does not ship a lockfile. Generate it only in the disposable
# extraction, then prove the test itself performs no dependency resolution.
env -u RUSTFLAGS CARGO_TARGET_DIR="$extract_target_dir" \
  cargo generate-lockfile --manifest-path "$extracted_manifest"
env -u RUSTFLAGS CARGO_TARGET_DIR="$extract_target_dir" \
  cargo test --manifest-path "$extracted_manifest" --all-features --locked
