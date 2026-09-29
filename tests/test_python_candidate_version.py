"""Candidate wheels must be distinct while release manifests stay strict."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


candidate_script = load_script("prepare_python_candidate")
release_metadata = load_script("validate_release_metadata")


def copy_manifests(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "Cargo.toml", tmp_path / "Cargo.toml")
    package = tmp_path / "crates" / "dag-ml-py"
    package.mkdir(parents=True)
    for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml"):
        shutil.copyfile(ROOT / "crates" / "dag-ml-py" / name, package / name)
    typed = package / "python" / "dag_ml"
    typed.mkdir(parents=True)
    (typed / "py.typed").write_text("", encoding="utf-8")
    (typed / "__init__.pyi").write_text("", encoding="utf-8")
    return tmp_path


def set_stable_python_version(root: Path) -> None:
    """Model a checkout immediately after a stable tag for the conversion test."""
    with (root / "Cargo.toml").open("rb") as handle:
        released = tomllib.load(handle)["workspace"]["package"]["version"]
    major, minor, patch = (int(part) for part in released.split("."))
    next_patch = f"{major}.{minor}.{patch + 1}"
    package = root / "crates" / "dag-ml-py"
    with (package / "Cargo.toml").open("rb") as handle:
        current = tomllib.load(handle)["package"]["version"]
    if current == released:
        return
    assert current == f"{next_patch}-dev.0"
    for name, old, new in (
        ("Cargo.toml", f'name = "dag-ml-py"\ndescription = "Python bindings for DAG-ML JSON contracts."\nversion = "{next_patch}-dev.0"',
         f'name = "dag-ml-py"\ndescription = "Python bindings for DAG-ML JSON contracts."\nversion = "{released}"'),
        ("Cargo.lock", f'name = "dag-ml-py"\nversion = "{next_patch}-dev.0"',
         f'name = "dag-ml-py"\nversion = "{released}"'),
        ("pyproject.toml", f'name = "dag-ml"\nversion = "{next_patch}.dev0"',
         f'name = "dag-ml"\nversion = "{released}"'),
    ):
        path = package / name
        path.write_text(candidate_script.replace_once(path.read_text(encoding="utf-8"), old, new, name), encoding="utf-8")


def test_next_patch_candidate_is_coherent_but_not_a_release(tmp_path: Path) -> None:
    root = copy_manifests(tmp_path)
    set_stable_python_version(root)
    with (root / "Cargo.toml").open("rb") as handle:
        released = tomllib.load(handle)["workspace"]["package"]["version"]
    major, minor, patch = (int(part) for part in released.split("."))
    next_patch = f"{major}.{minor}.{patch + 1}"
    native, wheel = candidate_script.prepare(root)
    assert (native, wheel) == (f"{next_patch}-dev.0", f"{next_patch}.dev0")
    package = root / "crates" / "dag-ml-py"
    with (package / "Cargo.lock").open("rb") as handle:
        locked = tomllib.load(handle)["package"]
    assert (
        next(item for item in locked if item["name"] == "dag-ml-py")["version"]
        == native
    )
    assert (
        next(item for item in locked if item["name"] == "dag-ml-core")["version"]
        == released
    )
    release_metadata.validate_python(root, "dag-ml", released)
    with pytest.raises(SystemExit, match="for a release"):
        release_metadata.validate_python(root, "dag-ml", released, release=True)


def test_candidate_preparation_is_idempotent_but_refuses_lock_drift(tmp_path: Path) -> None:
    root = copy_manifests(tmp_path)
    set_stable_python_version(root)
    prepared = candidate_script.prepare(root)
    assert candidate_script.prepare(root) == prepared

    root = copy_manifests(tmp_path / "second")
    set_stable_python_version(root)
    with (root / "Cargo.toml").open("rb") as handle:
        released = tomllib.load(handle)["workspace"]["package"]["version"]
    lock = root / "crates" / "dag-ml-py" / "Cargo.lock"
    lock.write_text(
        lock.read_text(encoding="utf-8").replace(
            f'name = "dag-ml-py"\nversion = "{released}"',
            'name = "dag-ml-py"\nversion = "0.0.0"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Cargo.lock"):
        candidate_script.prepare(root)


@pytest.mark.parametrize("name", ["Cargo.lock", "pyproject.toml"])
def test_checked_in_candidate_refuses_metadata_drift(tmp_path: Path, name: str) -> None:
    root = copy_manifests(tmp_path)
    native, wheel = candidate_script.prepare(root)
    path = root / "crates" / "dag-ml-py" / name
    source = path.read_text(encoding="utf-8")
    candidate = native if name == "Cargo.lock" else wheel
    path.write_text(source.replace(candidate, "0.0.0", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate metadata or Cargo.lock"):
        candidate_script.prepare(root)


def test_candidate_build_cannot_publish_without_a_release_tag() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-python.yml").read_text(
        encoding="utf-8"
    )
    publishing = workflow.split("  publish-pypi:\n", 1)[1].split(
        "  github-release:\n", 1
    )[0]
    assert "github.event_name != 'workflow_dispatch'" in publishing
    assert "python scripts/validate_release_metadata.py --release" in publishing
    assert '"${GITHUB_REF_TYPE:-}" != "tag"' in publishing
