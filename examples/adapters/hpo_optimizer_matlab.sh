#!/bin/sh
# Executable bridge for the MATLAB/Octave function used by --optimizer-adapter.
set -eu
DAGML_HPO_ADAPTER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export DAGML_HPO_ADAPTER_DIR
if command -v octave >/dev/null 2>&1; then
    exec octave --quiet --no-gui --no-init-file --eval 'addpath(getenv("DAGML_HPO_ADAPTER_DIR")); hpo_optimizer_jsonl'
fi
if command -v matlab >/dev/null 2>&1; then
    exec matlab -batch 'addpath(getenv("DAGML_HPO_ADAPTER_DIR")); hpo_optimizer_jsonl'
fi
echo 'hpo_optimizer_matlab.sh requires Octave or MATLAB' >&2
exit 127
