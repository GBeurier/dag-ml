from __future__ import annotations

import os
import subprocess
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the current workspace.
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "test_methods_optimizer_local.sh"
METHODS_RUNTIME_SOURCE_SHA = "4983c9a1df39d430a78c615bda209d3353514aa1"


def _probe(library_path: str | None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if library_path is None:
        env.pop("N4M_LIBRARY_PATH", None)
    else:
        env["N4M_LIBRARY_PATH"] = library_path
    return subprocess.run(
        [str(SCRIPT), "--probe"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_probe_requires_an_explicit_absolute_runtime_file(tmp_path: Path) -> None:
    missing = _probe(None)
    assert missing.returncode == 2
    assert "N4M_LIBRARY_PATH must explicitly name" in missing.stderr

    relative = _probe("libn4m.so")
    assert relative.returncode == 2
    assert "must be an absolute path" in relative.stderr

    runtime = tmp_path / "libn4m.so"
    runtime.write_bytes(b"test runtime")
    accepted = _probe(str(runtime))
    assert accepted.returncode == 0
    assert accepted.stdout == f"N4M_LIBRARY_PATH={runtime}\n"


def test_ci_and_local_selector_use_the_published_dynamic_binding() -> None:
    primary = tomllib.loads(
        (ROOT / "crates" / "dag-ml-core" / "Cargo.toml").read_text(encoding="utf-8")
    )
    dependency = primary["dependencies"]["n4m"]
    assert dependency["version"] == "0.1.2"
    assert dependency["default-features"] is False
    assert dependency["features"] == ["dynamic"]
    assert "git" not in dependency
    assert "path" not in dependency
    assert "methods-optimizer-local" not in primary["features"]
    assert not (ROOT / "crates" / "dag-ml-core" / "Cargo.toml.methods-local").exists()

    archive_core = primary["dev-dependencies"]["nirs4all_archive_core"]
    assert archive_core == {"package": "nirs4all", "version": "=0.3.22"}

    helper = SCRIPT.read_text(encoding="utf-8")
    assert "--features methods-optimizer" in helper
    assert 'feature=\\"methods-optimizer-local\\"' in helper

    python_lock = tomllib.loads(
        (ROOT / "crates" / "dag-ml-py" / "Cargo.lock").read_text(encoding="utf-8")
    )
    locked_n4m = [
        package for package in python_lock["package"] if package["name"] == "n4m"
    ]
    assert len(locked_n4m) == 1
    assert locked_n4m[0]["version"] == dependency["version"]
    assert locked_n4m[0]["source"] == (
        "registry+https://github.com/rust-lang/crates.io-index"
    )
    assert locked_n4m[0]["checksum"] == (
        "1bb1fcce4f16437c7ab55925fb05256e6f7e2f7a2e91a9d30b24805a19196dce"
    )

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    methods_job = workflow.split("  methods-hpo-local:", maxsplit=1)[1].split(
        "\n  sanitizer:", maxsplit=1
    )[0]
    assert f"ref: {METHODS_RUNTIME_SOURCE_SHA}" in methods_job
    assert "N4M_LIBRARY_PATH:" in methods_job
    assert "N4M_METHODS_REPO" not in methods_job
    assert "N4M_BINDING_SHA" not in methods_job

    assert (
        "python -m maturin build --release --locked --features extension-module"
        in workflow
    )
    release_workflow = (
        ROOT / ".github" / "workflows" / "release-python.yml"
    ).read_text(encoding="utf-8")
    assert "args: --release --locked --out dist" in release_workflow


def test_hpo_docs_match_the_published_fail_closed_feature_contract() -> None:
    primary = tomllib.loads(
        (ROOT / "crates" / "dag-ml-core" / "Cargo.toml").read_text(encoding="utf-8")
    )
    binding_version = primary["dependencies"]["n4m"]["version"]
    docs = (ROOT / "docs" / "HPO_METHODS_ADAPTER.md").read_text(encoding="utf-8")
    normalized_docs = " ".join(docs.split())

    assert "opt-in public `methods-optimizer` Cargo feature" in normalized_docs
    assert f"published dynamic `n4m` {binding_version} binding" in normalized_docs
    assert "Default builds leave that feature disabled and refuse HPO" in normalized_docs
    assert "no public `methods-optimizer` Cargo feature" not in normalized_docs
    assert "There is no sibling manifest or sibling source dependency" in normalized_docs
    assert '`nirs4all_archive_core = "=0.3.22"` development dependency' in normalized_docs
    assert "must not be presented as cross-source Core qualification" in normalized_docs
