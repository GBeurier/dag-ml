//! Process-description ingress through real campaign and HPO lifecycles.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn temp() -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "dagml-description-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&p).unwrap();
    p
}
fn base() -> serde_json::Value {
    serde_json::json!({"schema_version":1,"protocol":"dag-ml-process-adapter","adapter_id":"description-test",
        "supported_modes":["one_shot","jsonl"],"capabilities":["control_frames_v1","node_result_json_v1","node_task_json_v1","parallel_invocation_v1","persistent_workers","stateful_refit_artifacts","worker_env"]})
}
fn json(v: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec(v).unwrap()
}
fn extra(fragment: &[u8]) -> Vec<u8> {
    let mut b = json(&base());
    assert_eq!(b.pop(), Some(b'}'));
    b.push(b',');
    b.extend_from_slice(fragment);
    b.push(b'}');
    b
}
fn scenario(raw: &[u8], persistent: bool, parallel: bool, exit: Option<i32>) -> (Output, String) {
    let dir = temp();
    let description = dir.join("description.bin");
    let log = dir.join("events");
    fs::write(&description, raw).unwrap();
    let mut c = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"));
    c.current_dir(root())
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
            "crates/dag-ml-cli/tests/fixtures/process_description_ingress/ingress_adapter.py",
            "--plan-id",
            "plan:desc",
        ])
        .env("DESCFILE", description)
        .env("PROBELOG", &log)
        .env("PYTHONDONTWRITEBYTECODE", "1");
    if persistent {
        c.arg("--persistent");
    }
    if parallel {
        c.args([
            "--scheduler",
            "parallel",
            "--scheduler-workers",
            "2",
            "--process-workers",
            "2",
        ]);
    }
    if let Some(n) = exit {
        c.env("DEXIT", n.to_string()).env("DERR", "boom stderr");
    }
    let out = c.output().unwrap();
    let events = fs::read_to_string(&log).unwrap_or_default();
    #[cfg(target_os = "linux")]
    for line in events.lines().filter(|s| {
        s.starts_with("describe ") || s.starts_with("spawn ") || s.starts_with("oneshot ")
    }) {
        let pid = line.split_whitespace().nth(1).unwrap();
        assert!(!Path::new("/proc").join(pid).exists(), "PID remains: {pid}");
    }
    fs::remove_dir_all(dir).unwrap();
    (out, events)
}
fn refusal(raw: &[u8], persistent: bool, parallel: bool, expected: &str) {
    let (out, events) = scenario(raw, persistent, parallel, None);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "{stderr}");
    assert!(!String::from_utf8_lossy(&out.stdout).contains("process campaign run:"));
    assert!(stderr.contains(expected), "{stderr}");
    if expected == "process adapter description" {
        assert!(
            stderr.contains("describe handshake returned invalid JSON"),
            "{stderr}"
        );
        assert!(!stderr.contains("serialization error"), "{stderr}");
    }
    assert!(
        !events.lines().any(|s| s.starts_with("spawn ")
            || s.starts_with("oneshot ")
            || s.starts_with("recv task")),
        "{events}"
    );
    assert!(
        events.lines().any(|s| s.starts_with("describe ")),
        "{events}"
    );
}
fn success(raw: &[u8], persistent: bool) {
    let (out, events) = scenario(raw, persistent, false, None);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("8 result(s)") && stdout.contains("4 prediction block(s)"),
        "{stdout}"
    );
    assert!(
        events
            .lines()
            .any(|s| s.starts_with(if persistent { "recv task" } else { "oneshot " })),
        "{events}"
    );
}
#[test]
fn strict_description_and_preserved_capability_controls() {
    let d = base();
    let pos = serde_json::json!([
        1,
        d["protocol"],
        d["adapter_id"],
        d["supported_modes"],
        d["capabilities"]
    ]);
    let mut bad = vec![
        json(&pos),
        json(&serde_json::json!([
            1,
            "dag-ml-process-adapter",
            "x",
            ["one_shot"]
        ])),
        json(&serde_json::json!([
            1,
            "dag-ml-process-adapter",
            "x",
            ["one_shot"],
            ["node_task_json_v1", "node_result_json_v1"],
            "x"
        ])),
        extra(br#""x":1,"x":2"#),
        extra(br#""x":{"a":1,"a":2}"#),
        extra(br#""x":{"k":1,"\u006b":2}"#),
        extra(br#""\u00e9":1,"e\u0301":2"#),
        extra(br#""x":18446744073709551616"#),
        extra(br#""x":-9223372036854775809"#),
        extra(br#""x":1e400"#),
        extra(br#""x":"\ud800""#),
        extra(b"\"x\":\"\xff\""),
        extra(br#""protocol":"dag-ml-process-adapter""#),
        extra(br#""pr\u006ftocol":"dag-ml-process-adapter""#),
        extra(br#""x":"\q""#),
        b"\xff{}".to_vec(),
        b"{\n".to_vec(),
        b"null".to_vec(),
        b"   \n".to_vec(),
        b"\xef\xbb\xbf{}".to_vec(),
    ];
    let mut trailing = json(&d);
    trailing.extend_from_slice(b" {}");
    bad.push(trailing);
    for n in [127, 128, 129, 200_000] {
        let mut fragment = b"\"x\":".to_vec();
        fragment.extend(std::iter::repeat_n(b'[', n));
        fragment.extend(std::iter::repeat_n(b']', n));
        bad.push(extra(&fragment));
    }
    for raw in [
        br#""x":"ok""#.as_slice(),
        br#""x":{}"#,
        br#""x":18446744073709551615"#,
        br#""x":-9223372036854775808"#,
        br#""x":1e2"#,
        br#""x":0.1"#,
        br#""x":1e-400"#,
    ] {
        bad.push(extra(raw));
    }
    for (field, value) in [
        ("schema_version", serde_json::json!(null)),
        ("schema_version", serde_json::json!(1.0)),
        ("schema_version", serde_json::json!("1")),
        ("protocol", serde_json::json!(3)),
        ("supported_modes", serde_json::json!("one_shot")),
        ("supported_modes", serde_json::json!({"one_shot":true})),
        ("capabilities", serde_json::json!([["x"]])),
        ("capabilities", serde_json::json!([1])),
    ] {
        let mut v = d.clone();
        v[field] = value;
        bad.push(json(&v));
    }
    let mut missing = d.clone();
    missing.as_object_mut().unwrap().remove("capabilities");
    bad.push(json(&missing));
    for raw in bad {
        for persistent in [false, true] {
            refusal(&raw, persistent, false, "process adapter description");
        }
    }
    let mut controls = vec![json(&d), [json(&d), b"\r\n".to_vec()].concat()];
    for (field, value) in [
        ("adapter_id", serde_json::json!("bad id!")),
        (
            "supported_modes",
            serde_json::json!(["jsonl", "one_shot", "jsonl", "grpc"]),
        ),
        (
            "capabilities",
            serde_json::json!([
                "worker_env",
                "persistent_workers",
                "parallel_invocation_v1",
                "node_task_json_v1",
                "node_result_json_v1",
                "control_frames_v1",
                "node_task_json_v1",
                "",
                "bad cap!"
            ]),
        ),
    ] {
        let mut v = d.clone();
        v[field] = value;
        controls.push(json(&v));
    }
    for raw in controls {
        for persistent in [false, true] {
            success(&raw, persistent);
        }
    }
    let (parallel, _) = scenario(&json(&d), true, true, None);
    assert!(
        parallel.status.success(),
        "{}",
        String::from_utf8_lossy(&parallel.stderr)
    );
    assert!(String::from_utf8_lossy(&parallel.stdout).contains("8 result(s)"));
    for (field, value, expected) in [
        (
            "schema_version",
            serde_json::json!(2),
            "unsupported description schema version",
        ),
        (
            "protocol",
            serde_json::json!("other"),
            "unsupported protocol",
        ),
        ("adapter_id", serde_json::json!(" "), "empty adapter_id"),
        (
            "supported_modes",
            serde_json::json!(["one-shot", "JSONL"]),
            "does not support required mode",
        ),
    ] {
        let mut v = d.clone();
        v[field] = value;
        for persistent in [false, true] {
            refusal(&json(&v), persistent, false, expected);
        }
    }
    for cap in [
        "node_task_json_v1",
        "node_result_json_v1",
        "control_frames_v1",
        "persistent_workers",
        "worker_env",
        "parallel_invocation_v1",
    ] {
        let mut v = d.clone();
        v["capabilities"]
            .as_array_mut()
            .unwrap()
            .retain(|x| x.as_str() != Some(cap));
        let expected = if cap == "node_task_json_v1" || cap == "node_result_json_v1" {
            "missing required capability"
        } else if cap == "parallel_invocation_v1" {
            "missing required parallel scheduler capability"
        } else {
            "missing required persistent capability"
        };
        if cap.starts_with("node_") {
            for p in [false, true] {
                refusal(&json(&v), p, false, expected);
            }
        } else {
            refusal(&json(&v), true, cap == "parallel_invocation_v1", expected);
        }
    }
    for (mode, persistent) in [("one_shot", true), ("jsonl", false)] {
        let mut v = d.clone();
        v["supported_modes"] = serde_json::json!([mode]);
        refusal(
            &json(&v),
            persistent,
            false,
            "does not support required mode",
        );
    }
    for persistent in [false, true] {
        let (out, events) = scenario(b"", persistent, false, None);
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("returned empty stdout"));
        assert!(
            !events.contains("spawn ")
                && !events.contains("oneshot ")
                && !events.contains("recv task")
        );
        let (out, events) = scenario(&json(&d), persistent, false, Some(3));
        assert!(!out.status.success());
        assert!(String::from_utf8_lossy(&out.stderr).contains("boom stderr"));
        assert!(
            !events.contains("spawn ")
                && !events.contains("oneshot ")
                && !events.contains("recv task")
        );
    }
}

#[test]
fn invalid_description_keeps_label_through_host_hpo() {
    let dir = temp();
    let desc = dir.join("description");
    let log = dir.join("events");
    fs::write(&desc, extra(br#""unknown":1"#)).unwrap();
    let root = root();
    let fixtures = root.join("crates/dag-ml-cli/tests/fixtures/hpo_ingress");
    let capi = root.join("crates/dag-ml-capi/tests/fixtures/host_hpo_ingress");
    let out=Command::new(env!("CARGO_BIN_EXE_dag-ml-cli")).current_dir(&root).arg("run-host-hpo")
        .arg("--plan").arg(capi.join("plan.json"))
        .arg("--envelope").arg(root.join("crates/dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json"))
        .arg("--request").arg(fixtures.join("request.json"))
        .arg("--operator-adapter").arg(root.join("crates/dag-ml-cli/tests/fixtures/process_description_ingress/ingress_adapter.py"))
        .arg("--optimizer-adapter").arg(fixtures.join("opt.py"))
        .arg("--checkpoint").arg(dir.join("checkpoint.json")).arg("--output").arg(dir.join("output.json"))
        .env("DESCFILE",desc).env("PROBELOG",&log).env("DESC_HPO_DELEGATE","1").env("SC","control").env("PYTHONDONTWRITEBYTECODE","1").output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    let events = fs::read_to_string(log).unwrap_or_default();
    assert!(!out.status.success(), "{err}");
    assert!(err.contains("process adapter description"), "{err}");
    assert_eq!(
        err.matches("runtime validation failed:").count(),
        1,
        "{err}"
    );
    assert!(!events.contains("oneshot "), "{events}");
    fs::remove_dir_all(dir).unwrap();
}

fn r_available(package: &str) -> bool {
    let command=match package {"prospectr"=>"suppressPackageStartupMessages({library(jsonlite); library(prospectr)})", "mdatools"=>"suppressPackageStartupMessages({library(jsonlite); library(mdatools); library(digest)})", _=>unreachable!()};
    let ok = Command::new("Rscript")
        .current_dir(root())
        .args(["-e", command])
        .status()
        .map(|x| x.success())
        .unwrap_or(false);
    assert!(
        ok || std::env::var_os("DAG_ML_REQUIRE_R_ADAPTER_TEST").is_none(),
        "required R {package} adapter dependencies are unavailable"
    );
    ok
}

#[test]
fn real_description_bytes_replay_through_live_routes() {
    let adapters = [
        ("python_process_controller.py", None),
        ("flaky_process_controller.py", None),
        ("persisted_sidecar_process_controller.py", None),
        ("sklearn_process_controller.py", None),
        ("sklearn_production_controller.py", None),
        ("prospectr_process_controller.R", Some("prospectr")),
        ("mdatools_process_controller.R", Some("mdatools")),
    ];
    for (name, package) in adapters {
        if let Some(package) = package {
            if !r_available(package) {
                eprintln!("UNAVAILABLE: {name} description dependencies");
                continue;
            }
        }
        let mut c = Command::new(if package.is_some() {
            "Rscript"
        } else {
            "python3"
        });
        let output = c
            .current_dir(root())
            .arg(format!("examples/adapters/{name}"))
            .arg("--describe")
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let description: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let modes = description["supported_modes"].as_array().unwrap();
        for (mode, persistent) in [("one_shot", false), ("jsonl", true)] {
            if modes.iter().any(|v| v.as_str() == Some(mode)) {
                success(&output.stdout, persistent);
            }
        }
    }
}
