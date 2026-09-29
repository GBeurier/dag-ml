#!/usr/bin/env python3
"""Give a manual Python wheel build a distinct, unpublished next-patch version.

The Rust workspace stays at its tagged version. This script changes only the
excluded PyO3 crate and its Python package metadata in the disposable CI
checkout, so candidate wheels cannot be confused with the published wheel.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"{label}: expected exactly one {old!r}")
    return text.replace(old, new, 1)


def prepare(root: Path) -> tuple[str, str]:
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    released = workspace["workspace"]["package"]["version"]
    match = re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", released)
    if match is None:
        raise ValueError(
            f"workspace version must be a stable SemVer release: {released}"
        )
    base = f"{match[1]}.{match[2]}.{int(match[3]) + 1}"
    candidate_cargo = f"{base}-dev.0"
    candidate_python = f"{base}.dev0"

    package_dir = root / "crates" / "dag-ml-py"
    cargo_path = package_dir / "Cargo.toml"
    pyproject_path = package_dir / "pyproject.toml"
    lock_path = package_dir / "Cargo.lock"
    cargo = cargo_path.read_text(encoding="utf-8")
    pyproject = pyproject_path.read_text(encoding="utf-8")
    lock = lock_path.read_text(encoding="utf-8")

    if tomllib.loads(cargo)["package"]["version"] != released:
        raise ValueError(
            "Python Cargo package does not match the tagged workspace version"
        )
    if tomllib.loads(pyproject)["project"]["version"] != released:
        raise ValueError("Python pyproject does not match the tagged workspace version")
    locked = [
        package
        for package in tomllib.loads(lock)["package"]
        if package["name"] == "dag-ml-py"
    ]
    if len(locked) != 1 or locked[0]["version"] != released:
        raise ValueError(
            "Python Cargo.lock package does not match the tagged workspace version"
        )

    cargo = replace_once(
        cargo,
        f'name = "dag-ml-py"\ndescription = "Python bindings for DAG-ML JSON contracts."\nversion = "{released}"',
        f'name = "dag-ml-py"\ndescription = "Python bindings for DAG-ML JSON contracts."\nversion = "{candidate_cargo}"',
        str(cargo_path),
    )
    pyproject = replace_once(
        pyproject,
        f'name = "dag-ml"\nversion = "{released}"',
        f'name = "dag-ml"\nversion = "{candidate_python}"',
        str(pyproject_path),
    )
    lock = replace_once(
        lock,
        f'name = "dag-ml-py"\nversion = "{released}"',
        f'name = "dag-ml-py"\nversion = "{candidate_cargo}"',
        str(lock_path),
    )
    # Validate every transformed TOML document before writing any of them.
    for updated in (cargo, pyproject, lock):
        tomllib.loads(updated)
    cargo_path.write_text(cargo, encoding="utf-8")
    pyproject_path.write_text(pyproject, encoding="utf-8")
    lock_path.write_text(lock, encoding="utf-8")
    return candidate_cargo, candidate_python


if __name__ == "__main__":
    try:
        native, python = prepare(ROOT)
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        raise SystemExit(f"cannot prepare DAG-ML Python candidate: {exc}") from exc
    print(
        f"Prepared nonpublishing DAG-ML Python candidate: native={native}, wheel={python}"
    )
