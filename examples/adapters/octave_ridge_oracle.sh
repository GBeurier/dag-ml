#!/bin/sh
# Launch the real Octave Ridge operator for HPO, REFIT and replay.
set -eu
DAGML_OCTAVE_ADAPTER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export DAGML_OCTAVE_ADAPTER_DIR
if test "${1:-}" = --describe; then
    DAGML_OCTAVE_MODE=describe
else
    DAGML_OCTAVE_MODE=run
fi
export DAGML_OCTAVE_MODE
exec octave --quiet --no-gui --no-init-file --eval 'addpath(getenv("DAGML_OCTAVE_ADAPTER_DIR")); octave_ridge_oracle'
