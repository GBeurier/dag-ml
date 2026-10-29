#!/usr/bin/env python3
"""Build the ignored source extension from a measured, locked release wheel."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

from check_so_freshness import SO_RELATIVE, source_fingerprints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--python', default=os.environ.get('PYO3_PYTHON', 'python3'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.out.mkdir(parents=True, exist_ok=True)
    before = source_fingerprints(root)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    command = ['maturin', 'build', '--release', '--locked', '--manifest-path',
               'crates/dag-ml-py/Cargo.toml', '--out', str(args.out.resolve())]
    environment = os.environ.copy()
    environment['PYO3_PYTHON'] = args.python
    subprocess.run(command, cwd=root, env=environment, check=True)
    after = source_fingerprints(root)
    if before != after:
        raise RuntimeError('compiled source inputs changed during the wheel build')
    wheels = list(args.out.glob('dag_ml-*.whl'))
    if len(wheels) != 1:
        raise RuntimeError('a dedicated directory containing exactly one wheel is required')
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as archive:
        native = [name for name in archive.namelist() if name.endswith('.so')]
        if len(native) != 1:
            raise RuntimeError('the wheel must contain exactly one native extension')
        payload = archive.read(native[0])
    extension = root / SO_RELATIVE
    extension.write_bytes(payload)
    receipt = {
        'schema': 'dag-ml.native-source-build.v1', 'command': command,
        'cwd': str(root), 'source_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'started_at': started, 'finished_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'exit_code': 0, 'source_inputs_before': before, 'source_inputs_after': after,
        'wheel_path': str(wheel), 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
        'extension_sha256': hashlib.sha256(payload).hexdigest(),
        'environment': {key: environment.get(key) for key in ['PYO3_PYTHON', 'RUSTFLAGS', 'CARGO_HOME']},
    }
    target = root / 'target/python-source-build.json'
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps({'receipt': str(target), 'wheel_sha256': receipt['wheel_sha256'],
                      'extension_sha256': receipt['extension_sha256']}))


if __name__ == '__main__':
    main()
