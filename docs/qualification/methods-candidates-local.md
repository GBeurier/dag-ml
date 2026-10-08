# Local Methods candidate qualification

The Methods WASM and Octave candidate workflows build packages and check imports,
declarations, versions and MEX readiness. They publish nothing and produce no
numerical qualification capture. Full CV, HPO, refit, replay, archive and parity
checks run locally with the existing scripts and assertions below.

The build smokes compare the loaded Methods version's project and ABI components
with the checked-out header and package metadata. The version string includes
`+abi.`: the WASM default ref reports `1.2.1+abi.2.15.0`, while the historical
Octave library reports `1.2.1+abi.2.14.0`. Octave MEX files live in `+n4m` and use
qualified calls such as `n4m.n4m_version_mex()`. Both role and version MEX outputs
must be present before the build artifact is uploaded.

Use matching reviewed DAG, Methods, Core and SDK sources and dependencies. Build
the Node and web DAG packages into `target/wasm/dag-ml-wasm` and
`target/wasm-web/dag-ml-wasm`, and build/stage the Methods JS package into
`external/nirs4all-methods/bindings/js/dist`. These are the same paths used by the
candidate builds. Run from the DAG repository root and retain each command's
actual exit status, source/dependency identities and output files. Use a fresh
owned qualification directory; a successful build is not a numerical verdict.

## WASM and Python

The unchanged complete Node checks create the numerical capture used by the
Python and Octave comparisons. Run both Node and web controllers:

```sh
export QUALIFICATION_DIR=/absolute/path/to/new-owned-qualification
mkdir "$QUALIFICATION_DIR"
node scripts/smoke_wasm_bindings.cjs "$PWD/target/wasm/dag-ml-wasm"
node scripts/smoke_wasm_web_bindings.mjs "$PWD/target/wasm-web/dag-ml-wasm"
node scripts/smoke_wasm_n4m_hpo.mjs \
  "$PWD/target/wasm/dag-ml-wasm" \
  "$PWD/external/nirs4all-methods/bindings/js/dist"
node scripts/test_wasm_n4m_controller.mjs \
  "$PWD/target/wasm/dag-ml-wasm" \
  "$PWD/external/nirs4all-methods/bindings/js/dist"
node scripts/smoke_wasm_multimodal_methods_hpo.mjs \
  "$PWD/target/wasm/dag-ml-wasm" \
  "$PWD/external/nirs4all-methods/bindings/js/dist" \
  "$QUALIFICATION_DIR/four-source-receipt.json"
node scripts/test_wasm_n4m_controller.mjs \
  "$PWD/target/wasm-web/dag-ml-wasm" \
  "$PWD/external/nirs4all-methods/bindings/js/dist"
node scripts/smoke_wasm_multimodal_methods_hpo.mjs \
  "$PWD/target/wasm-web/dag-ml-wasm" \
  "$PWD/external/nirs4all-methods/bindings/js/dist"
python scripts/qualify_multimodal_methods_hpo_python.py \
  "$QUALIFICATION_DIR/four-source-receipt.json" \
  "$QUALIFICATION_DIR/python-parity.json"
```

The Python command requires the matching installed native DAG wheel and public
Methods runtime. The scripts retain their numeric comparisons, fold identity,
OOF, nested search, refit and replay assertions; no fixture or tolerance changes
are implied by moving their execution out of GitHub Actions.

## Octave and installed SDK archive consumer

Build the real Methods role/version MEX with matching native library and headers,
the native DAG CLI with `methods-optimizer`, and the matching Python wheel. Set
`N4M_ROLE_PIPELINE_JSON_FIXTURE` to the existing negative fixture belonging to the
same Methods source cohort. The build workflow's historical ABI 2.14 inputs stay
unchanged; use compatible refs when qualifying that historical cohort.

```sh
python external/nirs4all-methods/bindings/matlab/test/export_role_pipeline_fixture.py \
  "$N4M_ROLE_PIPELINE_JSON_FIXTURE" "$QUALIFICATION_DIR/octave-methods-fixture.m"
octave --no-gui --quiet --eval \
  "addpath('external/nirs4all-methods/bindings/matlab'); addpath('external/nirs4all-methods/bindings/matlab/test'); test_role_pipeline('$QUALIFICATION_DIR/octave-methods-fixture.m')"
export DAGML_REQUIRE_OCTAVE_ROLES=1
export DAG_ML_OCTAVE=/absolute/path/to/real/octave
export NIRS4ALL_OCTAVE_ROLE_NODE_CAPTURE="$QUALIFICATION_DIR/four-source-receipt.json"
cargo test -p dag-ml-core archive_workspace
python -m pytest tests/test_archive_v2_contract.py tests/test_octave_methods_role_adapter.py
python scripts/qualify_multimodal_methods_hpo_octave.py \
  --node-capture "$QUALIFICATION_DIR/four-source-receipt.json" \
  --cli target/release/dag-ml-cli --octave "$DAG_ML_OCTAVE" \
  --workdir "$QUALIFICATION_DIR/octave-qualification"
export NIRS4ALL_REQUIRE_OCTAVE_ROLE_CAMPAIGN=1
export NIRS4ALL_DAG_ML_ROOT="$PWD"
export DAG_ML_OCTAVE_MEX_PATH="$PWD/external/nirs4all-methods/bindings/matlab"
export PYTHONPATH="$PWD/external/nirs4all"
export NIRS4ALL_OCTAVE_ROLE_INSTALLED_PYTHON="$(python -c 'import sys; print(sys.executable)')"
python -m pytest external/nirs4all/tests/integration/api/test_octave_multimodal_role_archive.py \
  --basetemp="$QUALIFICATION_DIR/octave-sdk-proof"
```

Retain the source capture, proposals, five RAW model states, numerical parity,
fresh no-fit archive replay and refusal/lifecycle proofs produced by these
commands. The required-runtime flags remain enabled; missing Octave or native
dependencies must not turn the local campaign into a skipped qualification.
