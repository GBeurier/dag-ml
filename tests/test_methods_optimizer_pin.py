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
METHODS_RUNTIME_SOURCE_SHA = "aabdecfdd76d1a4d12cfbbade4eeee0d30a6ea47"


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


def test_ci_and_overlay_use_the_published_dynamic_binding() -> None:
    primary = tomllib.loads(
        (ROOT / "crates" / "dag-ml-core" / "Cargo.toml").read_text(encoding="utf-8")
    )
    overlay = tomllib.loads(
        (ROOT / "crates" / "dag-ml-core" / "Cargo.toml.methods-local").read_text(
            encoding="utf-8"
        )
    )
    for manifest in (primary, overlay):
        dependency = manifest["dependencies"]["n4m"]
        assert dependency["version"] == "0.1.1"
        assert dependency["default-features"] is False
        assert dependency["features"] == ["dynamic"]
        assert "git" not in dependency
        assert "path" not in dependency

    assert overlay["features"]["methods-optimizer-local"] == ["methods-optimizer"]

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    methods_job = workflow.split("  methods-hpo-local:", maxsplit=1)[1].split(
        "\n  sanitizer:", maxsplit=1
    )[0]
    assert f"ref: {METHODS_RUNTIME_SOURCE_SHA}" in methods_job
    assert "N4M_LIBRARY_PATH:" in methods_job
    assert "N4M_METHODS_REPO" not in methods_job
    assert "N4M_BINDING_SHA" not in methods_job
