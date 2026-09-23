#!/usr/bin/env python3
"""Refresh current W1 and D4 pack bytes after all code changes land.

Run this only after the final core/binding edits.  The base pack records exact
source bytes; D4 then pins that base pack and copies its artifact manifest.
Changing these in a different order leaves the replay pack internally stale.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parity.training import generate_fixtures as base
from parity.training import generate_training_replay_fixtures as replay


def _replace_assignment(path: Path, name: str, value: str) -> None:
    source = path.read_text()
    expression = rf'(?m)^({re.escape(name)} = ")[0-9a-f]{{64}}(")$'
    updated, count = re.subn(expression, lambda match: f"{match[1]}{value}{match[2]}", source)
    if count != 1:
        raise RuntimeError(f"expected one {name} assignment in {path}")
    if updated != source:
        path.write_text(updated)


def _replace_test_pin(path: Path, old: str, new: str) -> None:
    source = path.read_text()
    if source.count(f'"{old}"') != 1:
        raise RuntimeError(f"expected one old base-pack pin {old} in {path}")
    updated = source.replace(f'"{old}"', f'"{new}"')
    if updated != source:
        path.write_text(updated)


def main() -> None:
    # The published base pack is a snapshot of the current repository, not an
    # immutable release.  Its fixtures are maintained separately from this
    # byte-manifest refresh.
    old_sha = hashlib.sha256(base.PACK_PATH.read_bytes()).hexdigest()
    old_checksum = base.load_json(base.PACK_PATH)["pack_checksum"]
    base.generate_pack()
    new_sha = hashlib.sha256(base.PACK_PATH.read_bytes()).hexdigest()
    new_checksum = base.load_json(base.PACK_PATH)["pack_checksum"]

    generator = ROOT / "parity/training/generate_training_replay_fixtures.py"
    validator = ROOT / "scripts/validate_training_replay_contracts.py"
    tests = ROOT / "parity/training/tests/test_training_replay_contracts.py"
    for path in (generator, validator):
        _replace_assignment(path, "BASE_PACK_SHA256", new_sha)
        _replace_assignment(path, "BASE_PACK_CHECKSUM", new_checksum)
    # This assertion has its own literal pin, independent of both consumers.
    _replace_test_pin(tests, replay.BASE_PACK_SHA256, new_sha)
    _replace_test_pin(tests, replay.BASE_PACK_CHECKSUM, new_checksum)

    # The generator was imported before the pin files changed.  Set its live
    # constants so this process validates the same bytes as a fresh import.
    replay.BASE_PACK_SHA256 = new_sha
    replay.BASE_PACK_CHECKSUM = new_checksum
    replay.generate_pack()

    for command in (
        [sys.executable, "scripts/validate_contracts.py"],
        [sys.executable, "scripts/validate_training_replay_contracts.py"],
        [sys.executable, "-m", "pytest", "-q", "parity/training/tests/test_training_replay_contracts.py"],
    ):
        subprocess.run(command, cwd=ROOT, check=True)
    print(f"W1 base pack: {old_sha} → {new_sha}")
    print(f"W1 checksum: {old_checksum} → {new_checksum}")
    print(f"D4 replay pack: {hashlib.sha256(replay.PACK_PATH.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
