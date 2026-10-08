"""Candidate wheels must be distinct while release manifests stay strict."""

from __future__ import annotations

import importlib.util
import json
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
    with pytest.MonkeyPatch.context() as paths:
        paths.syspath_prepend(str(path.parent))
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
        (
            "Cargo.toml",
            f'name = "dag-ml-py"\ndescription = "Python bindings for DAG-ML JSON contracts."\nversion = "{next_patch}-dev.0"',
            f'name = "dag-ml-py"\ndescription = "Python bindings for DAG-ML JSON contracts."\nversion = "{released}"',
        ),
        (
            "Cargo.lock",
            f'name = "dag-ml-py"\nversion = "{next_patch}-dev.0"',
            f'name = "dag-ml-py"\nversion = "{released}"',
        ),
        (
            "pyproject.toml",
            f'name = "dag-ml"\nversion = "{next_patch}.dev0"',
            f'name = "dag-ml"\nversion = "{released}"',
        ),
    ):
        path = package / name
        path.write_text(
            candidate_script.replace_once(
                path.read_text(encoding="utf-8"), old, new, name
            ),
            encoding="utf-8",
        )


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


def test_candidate_preparation_is_idempotent_but_refuses_lock_drift(
    tmp_path: Path,
) -> None:
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


def test_ci_preserves_runtime_msrv_and_historical_oracle_targets() -> None:
    release_metadata.validate_ci(ROOT)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "run: cargo check --workspace --locked\n",
            "run: cargo check --workspace\n",
            "runtime targets",
        ),
        (
            "toolchain: ${{ env.RUST_ORACLE_TOOLCHAIN }}",
            "toolchain: ${{ env.RUST_MSRV }}",
            "oracle toolchain for all targets",
        ),
        (
            "run: cargo check --workspace --all-targets --locked\n",
            "run: cargo check --workspace --locked\n",
            "locked all-targets check",
        ),
        (
            'RUST_ORACLE_TOOLCHAIN: "1.88.0"',
            'RUST_ORACLE_TOOLCHAIN: "1.85.0"',
            "pin the historical Core oracle toolchain",
        ),
    ],
)
def test_ci_refuses_loss_of_locked_runtime_or_oracle_checks(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert workflow.count(old) == 1
    target = tmp_path / ".github" / "workflows"
    target.mkdir(parents=True)
    (target / "ci.yml").write_text(workflow.replace(old, new, 1))
    with pytest.raises(SystemExit, match=message):
        release_metadata.validate_ci(tmp_path)


def test_local_qualification_guards_precede_each_publication_writer() -> None:
    release_metadata.validate_local_publication_guards(ROOT)


@pytest.mark.parametrize(
    "filename,job",
    [
        ("release-python.yml", "publish-pypi"),
        ("release-python.yml", "github-release"),
        ("release-crates.yml", "publish-crates"),
        ("release-npm.yml", "build-and-publish"),
    ],
)
@pytest.mark.parametrize("mutation", ["remove", "disable", "continue", "move_after"])
def test_publication_refuses_bypassed_local_qualification(
    tmp_path: Path, filename: str, job: str, mutation: str
) -> None:
    target = tmp_path / ".github/workflows"
    target.mkdir(parents=True)
    for name in ["release-python.yml", "release-crates.yml", "release-npm.yml"]:
        shutil.copyfile(ROOT / ".github/workflows" / name, target / name)
    p = target / filename
    source = p.read_text()
    marker = "  " + job + ":\n"
    prefix, content = source.split(marker, 1)
    command = "        run: python scripts/verify_local_qualification.py --project dag --receipt compat/local-qualification.json --root ."
    assert command in content
    if mutation == "remove":
        content = content.replace(command, "        run: echo removed", 1)
    elif mutation == "disable":
        start = content.index(
            "      - name: Verify local runtime qualification before publication"
        )
        end = content.index(command, start)
        content = (
            content[:start]
            + content[start:end].replace("        if:", "        # old condition:")
            + "        if: false\n"
            + content[end:]
        )
    elif mutation == "continue":
        content = content.replace(
            command, "        continue-on-error: true\n" + command, 1
        )
    else:
        start = content.index(
            "      - name: Verify local runtime qualification before publication"
        )
        end = content.index(command, start) + len(command) + 1
        guard = content[start:end]
        content = content[:start] + content[end:] + guard
    p.write_text(prefix + marker + content)
    with pytest.raises(SystemExit, match="local qualification"):
        release_metadata.validate_local_publication_guards(tmp_path)


@pytest.mark.parametrize(
    "filename,job",
    [
        ("release-python.yml", "publish-pypi"),
        ("release-python.yml", "github-release"),
        ("release-crates.yml", "publish-crates"),
    ],
)
def test_writer_cannot_run_after_failed_local_qualification(
    tmp_path: Path, filename: str, job: str
) -> None:
    target = tmp_path / ".github/workflows"
    target.mkdir(parents=True)
    for name in ["release-python.yml", "release-crates.yml", "release-npm.yml"]:
        shutil.copyfile(ROOT / ".github/workflows" / name, target / name)
    p = target / filename
    prefix, content = p.read_text().split("  " + job + ":\n", 1)
    if job == "publish-pypi":
        content = content.replace(
            "      - uses: pypa/gh-action-pypi-publish@release/v1",
            "      - uses: pypa/gh-action-pypi-publish@release/v1\n        if: always()",
            1,
        )
    elif job == "github-release":
        content = content.replace(
            "      - uses: softprops/action-gh-release@v3",
            "      - uses: softprops/action-gh-release@v3\n        if: always()",
            1,
        )
    else:
        content = content.replace(
            "      - name: Publish workspace crates to crates.io",
            "      - name: Publish workspace crates to crates.io\n        if: always()",
            1,
        )
    p.write_text(prefix + "  " + job + ":\n" + content)
    with pytest.raises(SystemExit, match="publication condition"):
        release_metadata.validate_local_publication_guards(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "command", "count", "scope", "exclusion", "provenance", "host"],
)
def test_local_policy_preserves_extracted_package_coverage(
    tmp_path: Path, mutation: str
) -> None:
    target = tmp_path / ".github/workflows"
    target.mkdir(parents=True)
    for name in ["release-python.yml", "release-crates.yml", "release-npm.yml"]:
        shutil.copyfile(ROOT / ".github/workflows" / name, target / name)
    gate = {
        "id": "core-package-extract",
        "hosts": ["linux"],
        "commands": {"linux": ["bash", "scripts/test_core_package_extract.sh"]},
        "minimum_passed": 920,
        "input_paths": ["scripts/test_core_package_extract.sh"],
        "requires_provenance": True,
    }
    policy = {"gates": [gate]}
    path = tmp_path / "qualification/policy.json"
    path.parent.mkdir()
    path.write_text(json.dumps(policy))
    release_metadata.validate_local_publication_guards(tmp_path)
    if mutation == "missing":
        policy["gates"] = []
    elif mutation == "command":
        gate["commands"]["linux"] = ["echo", "passed"]
    elif mutation == "count":
        gate["minimum_passed"] = 0
    elif mutation == "scope":
        gate["input_paths"] = ["README.md"]
    elif mutation == "exclusion":
        gate["input_exclusions"] = ["scripts/test_core_package_extract.sh"]
    elif mutation == "provenance":
        gate["requires_provenance"] = False
    else:
        gate["hosts"] = ["windows"]
    path.write_text(json.dumps(policy))
    with pytest.raises(SystemExit, match="extracted Core package"):
        release_metadata.validate_local_publication_guards(tmp_path)
