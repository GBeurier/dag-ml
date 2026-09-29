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


def test_next_patch_candidate_is_coherent_but_not_a_release(tmp_path: Path) -> None:
    root = copy_manifests(tmp_path)
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


def test_candidate_preparation_refuses_retagging_or_lock_drift(tmp_path: Path) -> None:
    root = copy_manifests(tmp_path)
    candidate_script.prepare(root)
    with pytest.raises(ValueError, match="does not match"):
        candidate_script.prepare(root)

    root = copy_manifests(tmp_path / "second")
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
