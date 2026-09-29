#!/bin/sh
set -eu
if [ -z "${DAGML_N4M_PYTHON:-}" ]; then
    echo "DAGML_N4M_PYTHON must name a Python interpreter with n4m installed" >&2
    exit 2
fi
adapter_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$DAGML_N4M_PYTHON" "$adapter_dir/hpo_n4m_optimizer_jsonl.py"
