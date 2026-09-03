use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn cli() -> &'static str {
    env!("CARGO_BIN_EXE_dag-ml-cli")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("dag-ml-{label}-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create test directory");
        Self(path)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn marker_count(dir: &Path, prefix: &str) -> usize {
    std::fs::read_dir(dir)
        .expect("lifecycle marker directory")
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(prefix))
        .count()
}

#[test]
fn dag001_arrow_cache_replays_across_process_and_closes_controllers() {
    let root = workspace_root();
    let temp = TempDir::new("dag001-arrow-replay");
    let store = temp.0.join("cache");
    let lifecycle = temp.0.join("lifecycle");

    // Process A exports the signed Arrow IPC members. No in-memory payload or
    // controller state can cross the process boundary into the replay below.
    let export = Command::new(cli())
        .current_dir(&root)
        .args([
            "export-prediction-cache-store",
            "--bundle",
            "examples/generated/execution_bundle_branch_merge_cv_refit.json",
            "--payload",
            "examples/generated/prediction_cache_branch_merge_cv_refit.json",
            "--output-dir",
            store.to_str().expect("UTF-8 store path"),
            "--format",
            "arrow",
        ])
        .output()
        .expect("spawn Arrow cache exporter");
    assert!(
        export.status.success(),
        "Arrow cache export failed: {}",
        String::from_utf8_lossy(&export.stderr)
    );

    // Process B opens the manifest and IPC files, materializes both OOF cache
    // handles, hydrates all three replay artifact inputs, and drives persistent
    // external controllers. Their close acknowledgements prove that every
    // worker handle is released before this command returns.
    let replay = Command::new(cli())
        .current_dir(&root)
        .env("DAG_ML_PROCESS_LIFECYCLE_MARKER_DIR", &lifecycle)
        .args([
            "run-process-replay",
            "--bundle",
            "examples/generated/execution_bundle_branch_merge_cv_refit.json",
            "--graph",
            "examples/branch_merge_oof_graph.json",
            "--campaign",
            "examples/campaign_branch_merge_oof.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "branch:b0.model:ridge.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "branch:b1.model:rf.x=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--envelope",
            "merge:stack.pred_plus_original.meta:ridge.x_original=examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--replay-request",
            "examples/fixtures/bundle/replay_request_branch_merge_refit.json",
            "--prediction-cache-store",
            store.to_str().expect("UTF-8 store path"),
            "--adapter",
            "examples/adapters/python_process_controller.py",
            "--persistent",
            "--plan-id",
            "plan:generated.branch.merge.cv.refit",
            "--run-id",
            "run:dag001.arrow.fresh-process",
        ])
        .output()
        .expect("spawn fresh-process replay");
    assert!(
        replay.status.success(),
        "fresh-process replay failed: {}",
        String::from_utf8_lossy(&replay.stderr)
    );
    let stdout = String::from_utf8_lossy(&replay.stdout);
    assert!(
        stdout.contains("3 artifact handle(s)") && stdout.contains("2 prediction cache handle(s)"),
        "replay did not hydrate the complete artifact/cache surface: {stdout}"
    );
    let initialized = marker_count(&lifecycle, "init_");
    let closed = marker_count(&lifecycle, "close_");
    assert!(initialized > 0, "no production controller worker started");
    assert_eq!(closed, initialized, "not every controller worker closed");

    // The manifest fingerprints the physical IPC member in addition to the
    // semantic prediction fingerprint carried by Arrow metadata and bundle.
    let arrow_path = std::fs::read_dir(&store)
        .expect("read Arrow cache directory")
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .find(|path| {
            path.extension()
                .is_some_and(|extension| extension == "arrow")
        })
        .expect("Arrow cache member");
    let mut bytes = std::fs::read(&arrow_path).expect("read Arrow cache member");
    let last = bytes.last_mut().expect("non-empty Arrow cache member");
    *last ^= 1;
    std::fs::write(&arrow_path, bytes).expect("tamper Arrow cache member");
    let refused = Command::new(cli())
        .current_dir(&root)
        .args([
            "validate-prediction-cache-store",
            "--bundle",
            "examples/generated/execution_bundle_branch_merge_cv_refit.json",
            "--store-dir",
            store.to_str().expect("UTF-8 store path"),
        ])
        .output()
        .expect("spawn cache validator");
    assert!(
        !refused.status.success(),
        "tampered Arrow cache was accepted"
    );
    assert!(
        String::from_utf8_lossy(&refused.stderr).contains("IPC fingerprint"),
        "tamper refusal was not fingerprint-bound: {}",
        String::from_utf8_lossy(&refused.stderr)
    );
}
