"""The ignored source extension must match both its built wheel and Rust inputs."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('source_freshness', ROOT / 'scripts/check_so_freshness.py')
freshness = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(freshness)


def setup_artifact(tmp_path: Path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    rust = tmp_path / 'crates/dag-ml-core/src/lib.rs'
    rust.parent.mkdir(parents=True)
    rust.write_text('fn measured_source() {}\n')
    extension = tmp_path / freshness.SO_RELATIVE
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b'controlled native fixture')
    wheel = tmp_path / 'artifact.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('dag_ml/_dag_ml.abi3.so', extension.read_bytes())
    receipt = tmp_path / 'target/python-source-build.json'
    receipt.parent.mkdir()
    fingerprints = freshness.source_fingerprints(tmp_path)
    receipt.write_text(json.dumps({
        'schema': 'dag-ml.native-source-build.v1', 'exit_code': 0,
        'source_inputs_before': fingerprints, 'source_inputs_after': fingerprints,
        'extension_sha256': hashlib.sha256(extension.read_bytes()).hexdigest(),
        'wheel_path': str(wheel), 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }))
    return rust, extension, wheel, receipt


def test_ignored_extension_requires_successful_measured_build(tmp_path: Path):
    _, _, _, receipt = setup_artifact(tmp_path)
    assert freshness.check(tmp_path) == 0
    value = json.loads(receipt.read_text())
    value['exit_code'] = 1
    receipt.write_text(json.dumps(value))
    assert freshness.check(tmp_path) == 1
    receipt.unlink()
    assert freshness.check(tmp_path) == 1


def test_rust_source_change_invalidates_the_receipt(tmp_path: Path):
    rust, _, _, _ = setup_artifact(tmp_path)
    rust.write_text('fn changed_source() {}\n')
    assert freshness.check(tmp_path) == 1


def test_native_extension_tampering_invalidates_the_receipt(tmp_path: Path):
    _, extension, _, _ = setup_artifact(tmp_path)
    extension.write_bytes(b'different native fixture')
    assert freshness.check(tmp_path) == 1


def test_wheel_cannot_attest_different_native_bytes(tmp_path: Path):
    _, _, wheel, receipt = setup_artifact(tmp_path)
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('dag_ml/_dag_ml.abi3.so', b'different payload')
    value = json.loads(receipt.read_text())
    value['wheel_sha256'] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    receipt.write_text(json.dumps(value))
    assert freshness.check(tmp_path) == 1
