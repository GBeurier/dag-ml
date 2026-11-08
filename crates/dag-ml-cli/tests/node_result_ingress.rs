use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn campaign(case: &str, persistent: bool, retries: usize, once: bool) -> (Output, String) {
    let root = root();
    let dir = std::env::temp_dir().join(format!(
        "dagml-result-ingress-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).unwrap();
    let log = dir.join("events");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"));
    cmd.current_dir(&root)
        .args([
            "run-process-campaign",
            "--graph",
            "examples/minimal_graph.json",
            "--campaign",
            "examples/campaign_oof_generation.json",
            "--controllers",
            "examples/controller_manifests.json",
            "--envelope",
            "examples/fixtures/data/coordinator_data_plan_envelope_sample12.json",
            "--adapter",
            "crates/dag-ml-cli/tests/fixtures/node_result_ingress/operator.py",
            "--plan-id",
            "plan:probe",
        ])
        .env("SC", case)
        .env("PROBELOG", &log)
        .env("ONCE", if once { "1" } else { "0" })
        .env("PYTHONDONTWRITEBYTECODE", "1");
    if persistent {
        cmd.args(["--persistent", "--process-retries", &retries.to_string()]);
    }
    let output = cmd.output().unwrap();
    let events = fs::read_to_string(&log).unwrap_or_default();
    #[cfg(target_os = "linux")]
    for line in events
        .lines()
        .filter(|line| line.starts_with("spawn ") || line.starts_with("oneshot "))
    {
        let pid = line.split_whitespace().nth(1).unwrap();
        assert!(
            !Path::new("/proc").join(pid).exists(),
            "adapter {pid} still running"
        );
    }
    fs::remove_dir_all(dir).unwrap();
    (output, events)
}

fn refused(case: &str, persistent: bool, retries: usize, once: bool) {
    let (output, events) = campaign(case, persistent, retries, once);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "{case}: {stdout} {events}");
    assert!(
        !stdout.contains("process campaign run:"),
        "{case}: {stdout}"
    );
    let label = if persistent {
        "process adapter response frame"
    } else {
        "process adapter NodeResult"
    };
    assert!(stderr.contains(label), "{case}: {stderr}");
    assert!(stderr.contains("runtime validation"), "{case}: {stderr}");
    assert!(!stderr.contains("serialization error"), "{case}: {stderr}");
    if persistent && retries == 2 && once {
        assert_eq!(
            events.lines().filter(|s| s.starts_with("spawn ")).count(),
            2,
            "{case}: {events}"
        );
        assert_eq!(
            events
                .lines()
                .filter(|s| s.starts_with("recv task"))
                .count(),
            1,
            "{case}: {events}"
        );
    }
}

#[test]
fn strict_results_and_frames_use_real_process_lifecycle() {
    for case in [
        "dup",
        "oob_pos",
        "oob_neg",
        "opus_combined",
        "escaped_dup",
        "nfc_dup",
        "outputs_map_dup",
        "nested_dup",
        "top_dup",
        "unknown_field",
        "trailing",
        "invalid_utf8",
        "null_doc",
        "huge_exp",
        "metric_str",
    ] {
        for persistent in [false, true] {
            refused(case, persistent, 0, false);
        }
    }
    refused("positional_doc", false, 0, false);
    for case in [
        "ack_dup",
        "ack_type_dup",
        "ack_extra_dup",
        "ack_extra_oob",
        "ack_positional",
        "ack_null_version",
        "frame_outer_dup",
        "frame_positional",
        "error_retry_dup",
    ] {
        refused(case, true, 0, false);
    }
    for case in ["error_retry_positional", "error_retry_oob", "invalid_utf8"] {
        refused(case, true, 2, true);
    }
    for case in [
        "clean",
        "legal_u64max",
        "legal_i64min",
        "legal_exp",
        "schema_null",
    ] {
        for persistent in [false, true] {
            let (out, events) = campaign(case, persistent, 0, false);
            let stdout = String::from_utf8_lossy(&out.stdout);
            assert!(
                out.status.success(),
                "{case}: {} {events}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(
                stdout.contains("8 result(s)")
                    && stdout.contains("4 prediction block(s)")
                    && stdout.contains("4 data handle(s)"),
                "{stdout}"
            );
            if persistent {
                assert_eq!(
                    events
                        .lines()
                        .filter(|s| s.starts_with("recv task"))
                        .count(),
                    8
                );
                assert_eq!(
                    events
                        .lines()
                        .filter(|s| s.starts_with("recv init"))
                        .count(),
                    2
                );
                assert_eq!(events.lines().filter(|s| *s == "close").count(), 2);
            } else {
                assert_eq!(
                    events.lines().filter(|s| s.starts_with("oneshot ")).count(),
                    8
                );
            }
        }
    }
    for case in [
        "ack_extra",
        "frame_result_extra",
        "error_retry_clean",
        "eof_once",
    ] {
        let retry = matches!(case, "error_retry_clean" | "eof_once");
        let (out, events) = campaign(case, true, if retry { 2 } else { 0 }, retry);
        assert!(
            out.status.success(),
            "{case}: {} {events}",
            String::from_utf8_lossy(&out.stderr)
        );
        if retry {
            assert_eq!(
                events.lines().filter(|s| s.starts_with("spawn ")).count(),
                3
            );
            assert_eq!(
                events
                    .lines()
                    .filter(|s| s.starts_with("recv task"))
                    .count(),
                9
            );
        }
    }
}

fn python_has_sklearn(root: &Path) -> bool {
    let available = Command::new("python3")
        .current_dir(root)
        .args(["-c", "import sklearn, joblib"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    assert!(
        available || std::env::var_os("DAG_ML_REQUIRE_SKLEARN_TEST").is_none(),
        "required sklearn/joblib adapter dependencies are unavailable"
    );
    available
}

fn r_has_prospectr(root: &Path) -> bool {
    let available = Command::new("Rscript")
        .current_dir(root)
        .args([
            "-e",
            "suppressPackageStartupMessages({library(jsonlite); library(prospectr)})",
        ])
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    assert!(
        available || std::env::var_os("DAG_ML_REQUIRE_R_ADAPTER_TEST").is_none(),
        "required R prospectr adapter dependencies are unavailable"
    );
    available
}

fn r_has_mdatools(root: &Path) -> bool {
    let available = Command::new("Rscript")
        .current_dir(root)
        .args([
            "-e",
            "suppressPackageStartupMessages({library(jsonlite); library(mdatools); library(digest)})",
        ])
        .status()
        .map(|status| status.success())
        .unwrap_or(false);
    assert!(
        available || std::env::var_os("DAG_ML_REQUIRE_R_ADAPTER_TEST").is_none(),
        "required R mdatools adapter dependencies are unavailable"
    );
    available
}

#[test]
fn persistent_adapter_requires_control_frames_before_spawn() {
    let (out, events) = campaign("no_control_frames", true, 0, false);
    assert!(!out.status.success());
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("missing required persistent capability"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        events.lines().filter(|s| s.starts_with("spawn ")).count(),
        0
    );
}

#[test]
fn jsonl_adapters_reject_bare_tasks_and_close() {
    for adapter in [
        "python_process_controller.py",
        "sklearn_process_controller.py",
        "sklearn_production_controller.py",
        "flaky_process_controller.py",
        "mdatools_process_controller.R",
        "prospectr_process_controller.R",
    ] {
        let r = adapter.ends_with(".R");
        let available = match adapter {
            "sklearn_process_controller.py" | "sklearn_production_controller.py" => {
                python_has_sklearn(&root())
            }
            "prospectr_process_controller.R" => r_has_prospectr(&root()),
            "mdatools_process_controller.R" => r_has_mdatools(&root()),
            _ => true,
        };
        if !available {
            eprintln!("UNAVAILABLE: dependencies for {adapter}; adapter was not exercised");
            continue;
        }
        let mut child = Command::new(if r { "Rscript" } else { "python3" })
            .current_dir(root())
            .arg(format!("examples/adapters/{adapter}"))
            .arg("--jsonl")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let task =
            fs::read(root().join("examples/fixtures/runtime/node_task_transform_scale.json"))
                .unwrap();
        let mut stdin = child.stdin.take().unwrap();
        let task_value: serde_json::Value = serde_json::from_slice(&task).unwrap();
        writeln!(stdin, "{}", serde_json::to_string(&task_value).unwrap()).unwrap();
        writeln!(stdin, "{{\"type\":\"close\",\"schema_version\":1}}").unwrap();
        drop(stdin);
        let out = child.wait_with_output().unwrap();
        assert!(
            out.status.success(),
            "{adapter}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let rows: Vec<serde_json::Value> = String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(rows.len(), 2, "{adapter}");
        assert_eq!(rows[0]["type"], "error");
        assert_eq!(rows[0]["error"]["code"], "unsupported_frame");
        assert_eq!(rows[1]["type"], "ack");
        assert_eq!(rows[1]["status"], "closed");
    }
}
