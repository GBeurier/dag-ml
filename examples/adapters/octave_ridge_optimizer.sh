#!/bin/sh
set -eu
DAGML_OCTAVE_ADAPTER_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export DAGML_OCTAVE_ADAPTER_DIR
exec octave --quiet --no-gui --no-init-file --eval 'addpath(getenv("DAGML_OCTAVE_ADAPTER_DIR")); octave_ridge_optimizer'
