//! Exercise strict optimizer replies through the real CLI adapter workflow.
use std::{fs, path::PathBuf, process::Command};

#[test]
fn strict_optimizer_adapter_replies() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures/hpo_ingress");
    let capi = root.join("../dag-ml-capi/tests/fixtures/host_hpo_ingress");
    let envelope = root.join(
        "../dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json",
    );
    let dir = std::env::temp_dir().join(format!("dag-ml-cli-hpo-ingress-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    for (scenario, success) in [
        ("control", true),
        ("float-exponent", true),
        ("init-dup", false),
        ("ask-dup", false),
        ("ask-escape-dup", false),
        ("ask-oob", false),
        ("ask-nfc", false),
        ("extra-oob", false),
        ("tell-dup", false),
        ("prepare-dup", false),
        ("checkpoint-dup", false),
        ("error-dup", false),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
            .arg("run-host-hpo")
            .arg("--plan")
            .arg(capi.join("plan.json"))
            .arg("--envelope")
            .arg(&envelope)
            .arg("--request")
            .arg(fixtures.join("request.json"))
            .arg("--operator-adapter")
            .arg(fixtures.join("op.py"))
            .arg("--optimizer-adapter")
            .arg(fixtures.join("opt.py"))
            .arg("--checkpoint")
            .arg(dir.join(format!("{scenario}.checkpoint.json")))
            .arg("--output")
            .arg(dir.join(format!("{scenario}.out.json")))
            .env("SC", scenario)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.success(), success, "{scenario}: {stderr}");
        if !success {
            assert!(
                stderr.contains("HPO optimizer adapter reply"),
                "{scenario}: {stderr}"
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn typed_optimizer_members_preserve_shape_and_recovery() {
    use std::os::unix::fs::PermissionsExt;

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures/hpo_ingress");
    let capi = root.join("../dag-ml-capi/tests/fixtures/host_hpo_ingress");
    let envelope = root.join(
        "../dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json",
    );
    let dir = std::env::temp_dir().join(format!("dag-ml-cli-hpo-typed-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let wrapper = dir.join("operator.py");
    let operator = fixtures.join("op.py").canonicalize().unwrap();
    fs::write(&wrapper, format!("#!/usr/bin/env python3\nimport os, sys\nif sys.argv[1:] != ['--describe']:\n    open(os.environ['OPLOG'], 'a').write('fit\\n')\nos.execv({:?}, [{:?}] + sys.argv[1:])\n", operator.to_str().unwrap(), operator.to_str().unwrap())).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();

    let run =
        |scenario: &str, checkpoint: &std::path::Path, state: &std::path::Path, name: &str| {
            let log = dir.join(format!("{name}.fits"));
            let output_path = dir.join(format!("{name}.json"));
            fs::write(&log, "").unwrap();
            let output = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
                .arg("run-host-hpo")
                .arg("--plan")
                .arg(capi.join("plan.json"))
                .arg("--envelope")
                .arg(&envelope)
                .arg("--request")
                .arg(fixtures.join("request.json"))
                .arg("--operator-adapter")
                .arg(&wrapper)
                .arg("--optimizer-adapter")
                .arg(fixtures.join("opt.py"))
                .arg("--checkpoint")
                .arg(checkpoint)
                .arg("--output")
                .arg(&output_path)
                .env("SC", scenario)
                .env("STATE", state)
                .env("OPLOG", &log)
                .output()
                .unwrap();
            (output, fs::read_to_string(log).unwrap(), output_path)
        };
    let refusal = |output: &std::process::Output, fits: &str, label: &str| {
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(!output.status.success(), "{label}: {error}");
        assert!(
            error.contains("HPO optimizer adapter reply"),
            "{label}: {error}"
        );
        assert!(!error.contains("serialization error"), "{label}: {error}");
        assert!(error.contains("runtime validation"), "{label}: {error}");
        assert!(fits.is_empty(), "{label} invoked a fit: {fits}");
    };
    for scenario in [
        "params-array",
        "params-string",
        "params-number",
        "init-typed",
        "init-interrupted-typed",
    ] {
        let (output, fits, _) = run(
            scenario,
            &dir.join(format!("{scenario}.checkpoint")),
            &dir.join(format!("{scenario}.state")),
            scenario,
        );
        refusal(&output, &fits, scenario);
    }
    let (output, fits, path) = run(
        "params-null",
        &dir.join("null.checkpoint"),
        &dir.join("null.state"),
        "null",
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(fits.is_empty());
    let exhausted: serde_json::Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    assert_eq!(exhausted["status"], "exhausted");

    for (first, second, expect_success) in [
        ("tell-dup", "resume-prepared-positional", false),
        ("tell-dup", "resume-prepared", true),
        ("crash-at-ask1", "resume-interrupted-positional", false),
        ("crash-at-ask1", "resume-interrupted", true),
    ] {
        let checkpoint = dir.join(format!("{second}.checkpoint"));
        let state = dir.join(format!("{second}.state"));
        let (initial, _, _) = run(first, &checkpoint, &state, &format!("{second}-first"));
        assert!(!initial.status.success());
        assert!(
            checkpoint.is_file(),
            "missing owner checkpoint for {second}"
        );
        if first == "tell-dup" {
            assert!(state.is_file(), "missing owner prepared checkpoint");
        }
        let (output, fits, path) = run(second, &checkpoint, &state, second);
        if expect_success {
            assert!(
                output.status.success(),
                "{second}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let outcome: serde_json::Value =
                serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
            assert_eq!(outcome["status"], "completed");
        } else {
            refusal(&output, &fits, second);
        }
    }
}

#[cfg(unix)]
#[test]
fn unknown_evidence_refuses_saved_and_prepared_before_fit() {
    use std::os::unix::fs::PermissionsExt;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures/hpo_ingress");
    let capi = root.join("../dag-ml-capi/tests/fixtures/host_hpo_ingress");
    let envelope = root.join(
        "../dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json",
    );
    let dir = std::env::temp_dir().join(format!("dag-ml-cli-hpo-evidence-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let optimizer = fixtures.join("opt.py").canonicalize().unwrap();
    let wrapper = dir.join("optimizer.py");
    fs::write(&wrapper, format!("#!/usr/bin/env python3\nimport os, sys\nopen(os.environ['SPAWN'], 'a').write('spawned\\n')\nos.execv({:?}, [{:?}])\n", optimizer.to_str().unwrap(), optimizer.to_str().unwrap())).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();
    let operator = fixtures.join("op.py").canonicalize().unwrap();
    let operator_wrapper = dir.join("operator.py");
    fs::write(&operator_wrapper, format!("#!/usr/bin/env python3\nimport os, sys\nif sys.argv[1:] != ['--describe']:\n    open(os.environ['FITS'], 'a').write('fit\\n')\nos.execv({:?}, [{:?}] + sys.argv[1:])\n", operator.to_str().unwrap(), operator.to_str().unwrap())).unwrap();
    fs::set_permissions(&operator_wrapper, fs::Permissions::from_mode(0o755)).unwrap();
    let fits = dir.join("fits");
    let checkpoint = dir.join("owner.json");
    let state = dir.join("prepared.json");
    let events = dir.join("events");
    let spawned = dir.join("spawned");
    let run = |scenario: &str| {
        fs::write(&events, "").unwrap();
        fs::write(&spawned, "").unwrap();
        fs::write(&fits, "").unwrap();
        Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
            .args(["run-host-hpo", "--plan"])
            .arg(capi.join("plan.json"))
            .arg("--envelope")
            .arg(&envelope)
            .arg("--request")
            .arg(fixtures.join("request.json"))
            .arg("--operator-adapter")
            .arg(&operator_wrapper)
            .arg("--optimizer-adapter")
            .arg(&wrapper)
            .arg("--checkpoint")
            .arg(&checkpoint)
            .env("SC", scenario)
            .env("STATE", &state)
            .env("LOG", &events)
            .env("SPAWN", &spawned)
            .env("FITS", &fits)
            .output()
            .unwrap()
    };
    let initial = run("control");
    assert!(
        initial.status.success(),
        "{}",
        String::from_utf8_lossy(&initial.stderr)
    );
    let original = fs::read(&checkpoint).unwrap();
    let prepared = fs::read(&state).unwrap();
    for target in ["evidence", "scores", "report"] {
        for route in ["saved", "prepared"] {
            fs::write(&checkpoint, &original).unwrap();
            fs::write(&state, &prepared).unwrap();
            let path = if route == "saved" {
                &checkpoint
            } else {
                &state
            };
            let mut value: serde_json::Value =
                serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
            let evidence = &mut value["trials"][0]["evidence"];
            let field = match target {
                "evidence" => evidence,
                "scores" => &mut evidence["scores"],
                _ => &mut evidence["scores"]["reports"][0],
            };
            field["unknown_probe"] = serde_json::json!(true);
            let changed = serde_json::to_vec(&value).unwrap();
            fs::write(path, &changed).unwrap();
            let before_checkpoint = fs::read(&checkpoint).unwrap();
            let output = run(if route == "saved" {
                "control"
            } else {
                "resume-prepared"
            });
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(!output.status.success(), "{route} {target}: {error}");
            assert!(error.contains("unknown field"), "{error}");
            let log = fs::read_to_string(&events).unwrap();
            if route == "saved" {
                assert!(error.contains("failed to parse HPO checkpoint JSON"));
                assert!(log.is_empty(), "{log}");
            } else {
                assert!(error.contains("HPO optimizer adapter reply"));
                assert!(log.starts_with("init"));
                assert!(!log.contains("ask"));
            }
            assert_eq!(fs::read(&checkpoint).unwrap(), before_checkpoint);
            assert!(fs::read_to_string(&fits).unwrap().is_empty());
            // A rejected saved checkpoint may make the parent terminate the
            // spawned child before its first instruction can write the marker.
            let marker = fs::read_to_string(&spawned).unwrap();
            if route == "prepared" {
                assert_eq!(marker, "spawned\n");
            } else {
                assert!(marker.is_empty() || marker == "spawned\n");
            }
        }
    }
    fs::write(&state, &prepared).unwrap();
    let mut bad_seal: serde_json::Value = serde_json::from_slice(&original).unwrap();
    bad_seal["fingerprint"] = serde_json::json!("0".repeat(64));
    let bytes = serde_json::to_vec(&bad_seal).unwrap();
    fs::write(&checkpoint, &bytes).unwrap();
    let output = run("control");
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("fingerprint mismatch"));
    assert!(fs::read_to_string(&events).unwrap().starts_with("init"));
    assert!(fs::read_to_string(&fits).unwrap().is_empty());
    assert_eq!(fs::read(&checkpoint).unwrap(), bytes);
}

#[cfg(unix)]
#[test]
fn pruned_checkpoint_unknowns_refuse_before_init_or_ask() {
    use std::os::unix::fs::PermissionsExt;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures/hpo_ingress");
    let capi = root.join("../dag-ml-capi/tests/fixtures/host_hpo_ingress");
    let envelope = root.join(
        "../dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json",
    );
    let dir = std::env::temp_dir().join(format!(
        "dag-ml-cli-hpo-pruned-evidence-{}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();
    let request = dir.join("request.json");
    let mut request_value: serde_json::Value =
        serde_json::from_slice(&fs::read(fixtures.join("request.json")).unwrap()).unwrap();
    request_value["progressive_pruning"] = serde_json::json!(true);
    fs::write(&request, serde_json::to_vec(&request_value).unwrap()).unwrap();
    let optimizer = dir.join("optimizer.py");
    let source = fs::read_to_string(fixtures.join("opt.py")).unwrap();
    assert!(source.contains("r={'prune':False}"));
    fs::write(
        &optimizer,
        source.replace("r={'prune':False}", "r={'prune':True}"),
    )
    .unwrap();
    fs::set_permissions(&optimizer, fs::Permissions::from_mode(0o755)).unwrap();
    let checkpoint = dir.join("checkpoint.json");
    let prepared = dir.join("prepared.json");
    let events = dir.join("events");
    let run = |scenario: &str| {
        fs::write(&events, "").unwrap();
        Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
            .args(["run-host-hpo", "--plan"])
            .arg(capi.join("plan.json"))
            .arg("--envelope")
            .arg(&envelope)
            .arg("--request")
            .arg(&request)
            .arg("--operator-adapter")
            .arg(fixtures.join("op.py"))
            .arg("--optimizer-adapter")
            .arg(&optimizer)
            .arg("--checkpoint")
            .arg(&checkpoint)
            .env("SC", scenario)
            .env("STATE", &prepared)
            .env("LOG", &events)
            .output()
            .unwrap()
    };
    let owner = run("control");
    assert!(
        owner.status.success(),
        "{}",
        String::from_utf8_lossy(&owner.stderr)
    );
    let original = fs::read(&checkpoint).unwrap();
    let owner_prepared = fs::read(&prepared).unwrap();
    for route in ["saved", "prepared"] {
        for location in ["evidence", "scores", "report"] {
            fs::write(&checkpoint, &original).unwrap();
            fs::write(&prepared, &owner_prepared).unwrap();
            let target_file = if route == "saved" {
                &checkpoint
            } else {
                &prepared
            };
            let mut value: serde_json::Value =
                serde_json::from_slice(&fs::read(target_file).unwrap()).unwrap();
            assert_eq!(value["trials"][0]["state"], "pruned");
            let evidence = &mut value["trials"][0]["evidence"];
            let target = match location {
                "evidence" => evidence,
                "scores" => &mut evidence["scores"],
                _ => &mut evidence["scores"]["reports"][0],
            };
            target["unknown_probe"] = serde_json::json!(true);
            fs::write(target_file, serde_json::to_vec(&value).unwrap()).unwrap();
            let before = fs::read(&checkpoint).unwrap();
            let result = run(if route == "saved" {
                "control"
            } else {
                "resume-prepared"
            });
            let error = String::from_utf8_lossy(&result.stderr);
            assert!(!result.status.success(), "{route} {location}: {error}");
            assert!(error.contains("unknown field"), "{error}");
            let log = fs::read_to_string(&events).unwrap();
            if route == "saved" {
                assert!(error.contains("failed to parse HPO checkpoint JSON"));
                assert!(log.is_empty());
            } else {
                assert!(error.contains("HPO optimizer adapter reply"));
                assert!(log.starts_with("init"));
                assert!(!log.contains("ask"));
            }
            assert_eq!(fs::read(&checkpoint).unwrap(), before);
        }
    }
}

#[cfg(unix)]
#[test]
fn request_integer_spelling_is_checked_before_optimizer_spawn() {
    use std::os::unix::fs::PermissionsExt;
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixtures = root.join("tests/fixtures/hpo_ingress");
    let capi = root.join("../dag-ml-capi/tests/fixtures/host_hpo_ingress");
    let envelope = root.join(
        "../dag-ml-core/tests/fixtures/package/data/coordinator_data_plan_envelope_sample12.json",
    );
    let dir = std::env::temp_dir().join(format!("dag-ml-cli-hpo-numeric-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let optimizer = fixtures.join("opt.py").canonicalize().unwrap();
    let wrapper = dir.join("optimizer.py");
    fs::write(&wrapper, format!("#!/usr/bin/env python3\nimport os\nopen(os.environ['SPAWN'], 'a').write('spawned\\n')\nos.execv({:?}, [{:?}])\n", optimizer.to_str().unwrap(), optimizer.to_str().unwrap())).unwrap();
    fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755)).unwrap();
    let source = fs::read_to_string(fixtures.join("request.json")).unwrap();
    let needle = "\"owner\": \"capi-probe\"";
    assert!(source.contains(needle));
    for (index, (token, valid)) in [
        ("18446744073709551616", false),
        ("18446744073709551617", false),
        ("100000000000000000000", false),
        ("100000000000000000001", false),
        ("-9223372036854776000", false),
        ("100000000000000000000.0", true),
        ("1e+20", true),
        ("-0.0", true),
        ("5e-324", true),
        ("18446744073709551615", true),
        ("-9223372036854775808", true),
    ]
    .into_iter()
    .enumerate()
    {
        let request = dir.join(format!("request-{index}.json"));
        let marker = dir.join(format!("spawn-{index}"));
        let checkpoint = dir.join(format!("checkpoint-{index}.json"));
        fs::write(
            &request,
            source.replacen(needle, &format!("{needle}, \"probe\": {token}"), 1),
        )
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_dag-ml-cli"))
            .args(["run-host-hpo", "--plan"])
            .arg(capi.join("plan.json"))
            .arg("--envelope")
            .arg(&envelope)
            .arg("--request")
            .arg(&request)
            .arg("--operator-adapter")
            .arg(fixtures.join("op.py"))
            .arg("--optimizer-adapter")
            .arg(&wrapper)
            .arg("--checkpoint")
            .arg(&checkpoint)
            .env("SC", "control")
            .env("SPAWN", &marker)
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.success(), valid, "{token}: {error}");
        if valid {
            assert_eq!(fs::read_to_string(&marker).unwrap(), "spawned\n");
        } else {
            assert!(
                error.contains("HPO request") && error.contains("TCV1"),
                "{error}"
            );
            assert!(!marker.exists(), "optimizer spawned for {token}");
            assert!(!checkpoint.exists(), "checkpoint created for {token}");
        }
    }
}
