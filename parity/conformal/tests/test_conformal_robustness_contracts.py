"""Independent W0.6 JSON Schema, golden and semantic conformance tests."""

from __future__ import annotations

import ast
import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from parity.conformal.oracle import (  # noqa: E402
    ContractError,
    apply_json_pointer_mutation,
    assert_calibration_disjoint,
    file_sha256,
    fingerprint_without,
    load_json,
    regression_conformal_metrics,
    semantic_validator,
    split_absolute_residual,
    tcv1_preimage,
    tcv1_sha256,
    validate_calibration_artifact,
    validate_decision_block,
    validate_domain_assessment,
    validate_metric_set,
    validate_numeric_evidence,
    validate_prediction_block,
    validate_report,
    validate_scenario,
)
from parity.robustness_rng.oracle import (  # noqa: E402
    tcv1_preimage as robustness_rng_tcv1_preimage,
)
from parity.schema_dependencies import (  # noqa: E402
    SchemaDependencyError,
    missing_schema_dependencies,
    schema_dependency_closure,
)
from scripts import validate_contracts as production_contracts  # noqa: E402
from parity.conformal.generate_fixtures import (  # noqa: E402
    BASE_PACK_ARTIFACTS,
    PACK_ARTIFACTS,
    conformance_pack,
    generate,
)

SCHEMA_DIR = ROOT / "docs" / "contracts"
PACK_PATH = SCHEMA_DIR / "conformal_robustness_conformance_pack.v1.json"
ORACLE_GOLDEN = (
    ROOT / "parity" / "conformal" / "golden" / "split_absolute_residual.v1.json"
)
METRIC_GOLDEN = (
    ROOT / "parity" / "conformal" / "golden" / "regression_conformal_metrics.v1.json"
)
ROBUSTNESS_RNG_GOLDEN = (
    ROOT / "parity" / "robustness_rng" / "golden" / "philox4x32_10_counter.v1.json"
)
CALIBRATION_FIXTURE = (
    ROOT
    / "examples"
    / "fixtures"
    / "conformal"
    / "split_absolute_residual_physical_sample.v1.json"
)

FIXTURES = {
    "conformal_calibration.schema.json": ROOT
    / "examples"
    / "fixtures"
    / "conformal"
    / "calibration_artifacts.v1.json",
    "cohort_manifest.schema.json": ROOT
    / "examples"
    / "fixtures"
    / "conformal"
    / "cohort_manifest_roles.v1.json",
    "conformal_prediction_block.schema.json": ROOT
    / "examples"
    / "fixtures"
    / "conformal"
    / "conformal_prediction_blocks.v1.json",
    "conformal_metric_set.schema.json": ROOT
    / "examples"
    / "fixtures"
    / "conformal"
    / "conformal_metric_sets.v1.json",
    "domain_assessment_block.schema.json": ROOT
    / "examples"
    / "fixtures"
    / "conformal"
    / "domain_assessment_blocks.v1.json",
    "decision_block.schema.json": ROOT
    / "examples"
    / "fixtures"
    / "conformal"
    / "decision_blocks.v1.json",
    "robustness_scenario_spec.schema.json": ROOT
    / "examples"
    / "fixtures"
    / "robustness"
    / "robustness_scenarios.v1.json",
    "robustness_report.schema.json": ROOT
    / "examples"
    / "fixtures"
    / "robustness"
    / "robustness_reports.v1.json",
}


def _valid_document(schema_name: str, case_id: str) -> dict[str, Any]:
    fixture = load_json(FIXTURES[schema_name])
    return copy.deepcopy(
        next(
            case["document"] for case in fixture["valid_cases"] if case["id"] == case_id
        )
    )


def _refingerprint(document: dict[str, Any], field: str) -> None:
    document[field] = fingerprint_without(document, field)


def _refingerprint_calibration(document: dict[str, Any]) -> None:
    document["predictor_binding_fingerprint"] = tcv1_sha256(
        document["predictor_binding"]
    )
    _refingerprint(document, "checksum")


def _schemas_and_registry() -> tuple[dict[str, dict[str, Any]], Registry]:
    schemas: dict[str, dict[str, Any]] = {}
    resources: list[tuple[str, Resource[Any]]] = []
    for path in sorted(SCHEMA_DIR.glob("*.schema.json")):
        schema = load_json(path)
        schema_id = schema.get("$id")
        assert isinstance(schema_id, str), f"{path} has no $id"
        Draft202012Validator.check_schema(schema)
        schemas[path.name] = schema
        resources.append((schema_id, Resource.from_contents(schema)))
    return schemas, Registry().with_resources(resources)


@pytest.fixture(scope="module")
def schemas_and_registry() -> tuple[dict[str, dict[str, Any]], Registry]:
    """Load every local schema so external refs never use the network."""

    return _schemas_and_registry()


@pytest.mark.parametrize("schema_name", sorted(FIXTURES))
def test_new_schema_is_draft_2020_12_and_has_local_refs(
    schema_name: str,
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    schema = schemas[schema_name]
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    Draft202012Validator(schema, registry=registry)


@pytest.mark.parametrize("schema_name", sorted(FIXTURES))
def test_positive_fixtures_pass_schema_and_independent_semantics(
    schema_name: str,
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    fixture = load_json(FIXTURES[schema_name])
    assert fixture["schema"] == schema_name
    validator = Draft202012Validator(schemas[schema_name], registry=registry)
    semantic = semantic_validator(schema_name)
    ids: set[str] = set()
    for case in fixture["valid_cases"]:
        assert case["id"] not in ids
        ids.add(case["id"])
        validator.validate(case["document"])
        semantic(case["document"])


def _schema_error_text(errors: list[Any]) -> str:
    return "\n".join(
        f"/{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in errors
    )


@pytest.mark.parametrize("schema_name", sorted(FIXTURES))
def test_negative_fixtures_fail_closed_with_expected_cause(
    schema_name: str,
    schemas_and_registry: tuple[dict[str, dict[str, Any]], Registry],
) -> None:
    schemas, registry = schemas_and_registry
    fixture = load_json(FIXTURES[schema_name])
    positive = {case["id"]: case["document"] for case in fixture["valid_cases"]}
    validator = Draft202012Validator(schemas[schema_name], registry=registry)
    semantic = semantic_validator(schema_name)
    fingerprint_field = {
        "conformal_calibration.schema.json": "checksum",
        "cohort_manifest.schema.json": "manifest_fingerprint",
        "conformal_prediction_block.schema.json": "block_fingerprint",
        "conformal_metric_set.schema.json": "metric_set_fingerprint",
        "domain_assessment_block.schema.json": "block_fingerprint",
        "decision_block.schema.json": "block_fingerprint",
        "robustness_scenario_spec.schema.json": "scenario_fingerprint",
        "robustness_report.schema.json": "report_fingerprint",
    }[schema_name]
    for case in fixture["invalid_cases"]:
        document = positive[case["base_case"]]
        for mutation in case["mutations"]:
            document = apply_json_pointer_mutation(
                document, mutation["path"], mutation["value"]
            )
        recompute_fingerprints = case.get("recompute_fingerprints") is True
        targets_fingerprint = case.get("targets_fingerprint") is True
        assert recompute_fingerprints != targets_fingerprint, case["id"]
        expected_fingerprint = fingerprint_without(document, fingerprint_field)
        if recompute_fingerprints:
            document[fingerprint_field] = fingerprint_without(
                document, fingerprint_field
            )
            assert document[fingerprint_field] == expected_fingerprint
        else:
            assert document[fingerprint_field] != expected_fingerprint
        schema_errors = list(validator.iter_errors(document))
        semantic_error = ""
        try:
            semantic(document)
        except ContractError as exc:
            semantic_error = str(exc)
        combined = f"{_schema_error_text(schema_errors)}\n{semantic_error}".lower()
        assert schema_errors or semantic_error, f"{case['id']} unexpectedly passed"
        assert case["expected_error"].lower() in combined, (
            f"{case['id']} failed for the wrong cause; expected {case['expected_error']!r}, "
            f"got {combined!r}"
        )


def test_cohort_roles_and_leakage_closure() -> None:
    fixture = load_json(FIXTURES["cohort_manifest.schema.json"])
    cohorts = {case["id"]: case["document"] for case in fixture["valid_cases"]}
    assert set(cohorts) == {"development", "calibration", "external_test", "production"}
    for case in fixture["disjointness_cases"]:
        action = lambda: assert_calibration_disjoint(  # noqa: E731
            case["training_sample_ids"],
            case["training_origin_sample_ids"],
            cohorts[case["calibration_case"]],
        )
        if case.get("expected") == "valid":
            action()
        else:
            with pytest.raises(ContractError, match=case["expected_error"]):
                action()


def test_split_absolute_residual_oracle_and_small_n_policy() -> None:
    fixture = load_json(ORACLE_GOLDEN)
    assert fixture["numeric_version"] == "split_absolute_residual.v1"
    for case in fixture["cases"]:
        action = lambda: split_absolute_residual(  # noqa: E731
            case["residuals"],
            case["coverages"],
            multi_target_policy=case["multi_target_policy"],
            small_sample_policy=case["small_sample_policy"],
        )
        if "expected_error" in case:
            with pytest.raises(ContractError, match=case["expected_error"]):
                action()
        else:
            assert action() == case["expected"]


def test_physical_sample_oracle_cases_match_independent_oracle() -> None:
    fixture = load_json(CALIBRATION_FIXTURE)
    for case in fixture["oracle_cases"]:
        if "expected_error" in case:
            with pytest.raises(ContractError):
                split_absolute_residual(
                    case["residuals"],
                    case["coverages"],
                    multi_target_policy="marginal",
                    small_sample_policy=case["small_sample_policy"],
                )
            continue
        records = split_absolute_residual(
            case["residuals"],
            case["coverages"],
            multi_target_policy="marginal",
            small_sample_policy=case["small_sample_policy"],
        )
        projected = [
            {
                "coverage": record["coverage"],
                "rank": record["rank"],
                "quantile": record["values"][0],
            }
            for record in records
        ]
        assert projected == case["expected"], case["id"]


def test_small_n_unbounded_wire_shape_is_exact() -> None:
    records = split_absolute_residual(
        [[0.1], [0.2], [0.3]],
        [0.95],
        multi_target_policy="marginal",
        small_sample_policy="unbounded",
    )
    assert records[0]["values"] == [{"status": "unbounded"}]
    assert set(records[0]["values"][0]) == {"status"}


def test_calibration_quantiles_are_binary64_and_monotone() -> None:
    base = load_json(CALIBRATION_FIXTURE)["calibration_artifact"]

    integer_quantile = copy.deepcopy(base)
    integer_quantile["quantiles"][0]["values"][0]["value"] = 1
    _refingerprint(integer_quantile, "checksum")
    with pytest.raises(ContractError, match="binary64"):
        validate_calibration_artifact(integer_quantile)

    decreasing = copy.deepcopy(base)
    first = decreasing["quantiles"][0]["values"][0]["value"]
    decreasing["quantiles"][1]["values"][0]["value"] = first - 0.1
    _refingerprint(decreasing, "checksum")
    with pytest.raises(ContractError, match="monotone"):
        validate_calibration_artifact(decreasing)


def test_wire_numeric_fields_refuse_integer_tokens() -> None:
    prediction = _valid_document(
        "conformal_prediction_block.schema.json", "marginal_two_target_nested"
    )
    prediction["intervals"][0]["lower"][0][0] = int(
        prediction["intervals"][0]["lower"][0][0]
    )
    _refingerprint(prediction, "block_fingerprint")
    with pytest.raises(ContractError, match="binary64"):
        validate_prediction_block(prediction)

    metric_set = _valid_document(
        "conformal_metric_set.schema.json", "marginal_two_target_metrics"
    )
    metric_set["records"][0]["empirical_coverage"] = 1
    _refingerprint(metric_set, "metric_set_fingerprint")
    with pytest.raises(ContractError, match="binary64"):
        validate_metric_set(metric_set)

    domain = _valid_document(
        "domain_assessment_block.schema.json", "two_unit_support_assessment"
    )
    domain["assessments"][0]["methods"][0]["score"] = int(
        domain["assessments"][0]["methods"][0]["score"]
    )
    _refingerprint(domain, "block_fingerprint")
    with pytest.raises(ContractError, match="binary64"):
        validate_domain_assessment(domain)
    with pytest.raises(production_contracts.ContractError, match="binary64"):
        production_contracts.validate_w06_domain_assessment(domain)

    decision = _valid_document("decision_block.schema.json", "accept_and_refer")
    decision["thresholds"][1]["value"] = int(decision["thresholds"][1]["value"])
    _refingerprint(decision, "block_fingerprint")
    with pytest.raises(ContractError, match="binary64"):
        validate_decision_block(decision)
    with pytest.raises(production_contracts.ContractError, match="binary64"):
        production_contracts.validate_w06_decision_block(decision)

    decision_membership = _valid_document(
        "decision_block.schema.json", "accept_and_refer"
    )
    decision_membership["thresholds"][0]["value"][0] = 1
    _refingerprint(decision_membership, "block_fingerprint")
    with pytest.raises(ContractError, match="binary64"):
        validate_decision_block(decision_membership)
    with pytest.raises(production_contracts.ContractError, match="binary64"):
        production_contracts.validate_w06_decision_block(decision_membership)

    scenario = _valid_document("robustness_scenario_spec.schema.json", "clean_frozen")
    scenario["severities"][0] = 0
    _refingerprint(scenario, "scenario_fingerprint")
    with pytest.raises(ContractError, match="binary64"):
        validate_scenario(scenario)

    report = _valid_document(
        "robustness_report.schema.json", "three_modes_resolved_conformal"
    )
    report["results"][0]["confidence_intervals"][0]["lower"] = 0
    _refingerprint(report, "report_fingerprint")
    with pytest.raises(ContractError, match="binary64"):
        validate_report(report)


def test_predictor_binding_closure_order_and_relation_are_audited() -> None:
    base = load_json(CALIBRATION_FIXTURE)["calibration_artifact"]
    predictor = base["predictor_binding"]
    assert "branch:b1.augment:noise" in predictor["predictor_node_ids"]

    mutations = (
        (
            "sorted by requirement_key",
            lambda artifact: artifact["predictor_binding"]["data_bindings"].reverse(),
        ),
        (
            "sorted by node_id",
            lambda artifact: artifact["predictor_binding"]["artifacts"].reverse(),
        ),
        (
            "relation_fingerprint",
            lambda artifact: artifact["predictor_binding"]["data_bindings"][0].update(
                relation_fingerprint="0" * 64
            ),
        ),
        (
            "authoritative predictor closure",
            lambda artifact: artifact["predictor_binding"]["predictor_node_ids"].remove(
                "branch:b1.augment:noise"
            ),
        ),
    )
    for expected_error, mutate in mutations:
        artifact = copy.deepcopy(base)
        mutate(artifact)
        _refingerprint_calibration(artifact)
        with pytest.raises(ContractError, match=expected_error):
            validate_calibration_artifact(artifact)


def test_scenario_kind_and_identity_severity_are_biconditional() -> None:
    clean = _valid_document("robustness_scenario_spec.schema.json", "clean_frozen")
    clean["perturbation"]["kind"] = "node_replacement"
    _refingerprint(clean, "scenario_fingerprint")
    with pytest.raises(ContractError, match="if and only if"):
        validate_scenario(clean)

    identity = _valid_document("robustness_scenario_spec.schema.json", "clean_frozen")
    identity["perturbation"]["kind"] = "identity"
    identity["rng"]["target_kind"] = "global"
    _refingerprint(identity, "scenario_fingerprint")
    with pytest.raises(ContractError, match=r"exactly severities \[0.0\]"):
        validate_scenario(identity)

    structural = _valid_document(
        "robustness_scenario_spec.schema.json", "structural_refit"
    )
    structural["perturbation"]["kind"] = "gaussian_noise"
    _refingerprint(structural, "scenario_fingerprint")
    with pytest.raises(ContractError, match="if and only if"):
        validate_scenario(structural)


def test_regression_conformal_metrics_reconstruct_from_truth_and_bounds() -> None:
    fixture = load_json(METRIC_GOLDEN)
    assert fixture["numeric_version"] == "split_absolute_residual.metrics.v1"
    assert {case["multi_target_policy"] for case in fixture["cases"]} == {
        "marginal",
        "joint_max",
    }
    for case in fixture["cases"]:
        assert (
            regression_conformal_metrics(
                case["truth"],
                case["interval"],
                multi_target_policy=case["multi_target_policy"],
            )
            == case["expected"]
        )


def test_numeric_evidence_closes_points_bounds_truth_and_metrics() -> None:
    prediction_fixture = load_json(FIXTURES["conformal_prediction_block.schema.json"])
    metric_fixture = load_json(FIXTURES["conformal_metric_set.schema.json"])
    validate_numeric_evidence(
        metric_fixture["evidence_cases"],
        [case["document"] for case in prediction_fixture["valid_cases"]],
        [case["document"] for case in metric_fixture["valid_cases"]],
        label="standalone numeric evidence",
    )

    report_fixture = load_json(FIXTURES["robustness_report.schema.json"])
    reports = {case["id"]: case["document"] for case in report_fixture["valid_cases"]}
    evidence_sets = {
        evidence_set["report_case"]: evidence_set["records"]
        for evidence_set in report_fixture["evidence_sets"]
    }
    assert set(evidence_sets) == {
        case_id
        for case_id, report in reports.items()
        if report["conformal_prediction_blocks"]
    }
    for case_id, evidence in evidence_sets.items():
        report = reports[case_id]
        validate_numeric_evidence(
            evidence,
            report["conformal_prediction_blocks"],
            report["conformal_metric_sets"],
            label=f"report numeric evidence {case_id}",
        )


def test_refingerprinted_numeric_evidence_tampering_fails_at_reconstruction() -> None:
    fixture = load_json(FIXTURES["robustness_report.schema.json"])
    reports = {case["id"]: case["document"] for case in fixture["valid_cases"]}
    evidence_sets = {
        evidence_set["report_case"]: evidence_set["records"]
        for evidence_set in fixture["evidence_sets"]
    }
    for case in fixture["invalid_evidence_cases"]:
        report = copy.deepcopy(reports[case["report_case"]])
        evidence = copy.deepcopy(evidence_sets[case["report_case"]])
        for mutation in case.get("report_mutations", []):
            report = apply_json_pointer_mutation(
                report, mutation["path"], mutation["value"]
            )
        evidence_index = next(
            index
            for index, record in enumerate(evidence)
            if record["evidence_id"] == case["base_evidence_id"]
        )
        mutated_record = evidence[evidence_index]
        for mutation in case["mutations"]:
            mutated_record = apply_json_pointer_mutation(
                mutated_record, mutation["path"], mutation["value"]
            )
        evidence[evidence_index] = mutated_record
        if case.get("rebind_metric_truth", False):
            metric_set = next(
                metric_set
                for metric_set in report["conformal_metric_sets"]
                if metric_set["metric_set_id"] == mutated_record["metric_set_id"]
            )
            metric_set["truth_fingerprint"] = mutated_record["truth_fingerprint"]
            _refingerprint(metric_set, "metric_set_fingerprint")
            _refingerprint(report, "report_fingerprint")
        validate_report(report)
        with pytest.raises(ContractError, match=case["expected_error"]):
            validate_numeric_evidence(
                evidence,
                report["conformal_prediction_blocks"],
                report["conformal_metric_sets"],
                label=case["id"],
            )


def test_tcv1_preimage_vectors_include_utf8_utf16_discriminator() -> None:
    fixture = load_json(ORACLE_GOLDEN)
    vectors = {vector["id"]: vector for vector in fixture["tcv1_vectors"]}
    assert "utf8_key_order_differs_from_utf16" in vectors
    for vector in vectors.values():
        assert tcv1_preimage(vector["value"]).hex() == vector["expected_preimage_hex"]
        assert tcv1_sha256(vector["value"]) == vector["expected_sha256"]
        assert tcv1_preimage(vector["equivalent_value"]) == tcv1_preimage(
            vector["value"]
        )


def test_conformal_and_rng_oracles_agree_on_every_tcv1_vector() -> None:
    conformal_vectors = load_json(ORACLE_GOLDEN)["tcv1_vectors"]
    for vector in conformal_vectors:
        expected = bytes.fromhex(vector["expected_preimage_hex"])
        assert tcv1_preimage(vector["value"]) == expected
        assert robustness_rng_tcv1_preimage(vector["value"]) == expected

    rng_vectors = load_json(ROBUSTNESS_RNG_GOLDEN)["vectors"]
    for vector in rng_vectors:
        payload = vector["expected"]["normalized_payload"]
        expected = bytes.fromhex(vector["expected"]["tcv1_preimage_hex"])
        assert tcv1_preimage(payload) == expected
        assert robustness_rng_tcv1_preimage(payload) == expected


def test_cross_fixture_cohort_and_scenario_identity_is_exact() -> None:
    cohorts = load_json(FIXTURES["cohort_manifest.schema.json"])["valid_cases"]
    external = next(
        case["document"] for case in cohorts if case["id"] == "external_test"
    )
    scenarios = load_json(FIXTURES["robustness_scenario_spec.schema.json"])[
        "valid_cases"
    ]
    scenario_documents = [case["document"] for case in scenarios]
    report = load_json(FIXTURES["robustness_report.schema.json"])["valid_cases"][0][
        "document"
    ]
    assert report["cohort_manifest"] == external
    assert report["scenarios"] == scenario_documents


def test_external_report_covers_three_modes_with_exact_baselines() -> None:
    fixture = load_json(FIXTURES["robustness_report.schema.json"])
    report = next(
        case["document"]
        for case in fixture["valid_cases"]
        if case["id"] == "three_modes_resolved_conformal"
    )

    assert len(report["calibration_artifacts"]) == 3
    assert len(report["conformal_prediction_blocks"]) == 18
    assert len(report["conformal_metric_sets"]) == 18
    assert len(report["results"]) == 18
    assert {scenario["mode"] for scenario in report["scenarios"]} == {
        "clean_frozen",
        "matched_recalibration",
        "structural_refit",
    }

    coordinates: dict[tuple[str, str, str | None], list[dict[str, Any]]] = {}
    for result in report["results"]:
        assert isinstance(result["severity"], float)
        key = (
            result["scenario_id"],
            result["slice"]["kind"],
            result["slice"]["value"],
        )
        coordinates.setdefault(key, []).append(result)

    scenario_severities = {
        (result["scenario_id"], result["severity"]) for result in report["results"]
    }
    expected_coordinates = {
        (scenario_id, severity, slice_kind, slice_value)
        for scenario_id, severities, slices in (
            (
                "scenario:clean.noise",
                (0.0, 0.01),
                (
                    ("all", None),
                    ("group", "group:batch.A"),
                    ("group", "group:batch.B"),
                    ("source", "nir"),
                    ("source", "nir.secondary"),
                ),
            ),
            (
                "scenario:matched.noise",
                (0.0, 0.5),
                (
                    ("all", None),
                    ("group", "group:batch.A"),
                    ("group", "group:batch.B"),
                ),
            ),
            (
                "scenario:structural.node",
                (0.0, 1.0),
                (("all", None),),
            ),
        )
        for severity in severities
        for slice_kind, slice_value in slices
    }
    assert {
        (
            result["scenario_id"],
            result["severity"],
            result["slice"]["kind"],
            result["slice"]["value"],
        )
        for result in report["results"]
    } == expected_coordinates
    for scenario_id, severity in scenario_severities:
        assert any(
            result["scenario_id"] == scenario_id
            and result["severity"] == severity
            and result["slice"] == {"kind": "all", "value": None}
            for result in report["results"]
        )

    for result in report["results"]:
        if result["severity"] == 0.0:
            continue
        key = (
            result["scenario_id"],
            result["slice"]["kind"],
            result["slice"]["value"],
        )
        baselines = [item for item in coordinates[key] if item["severity"] == 0.0]
        assert len(baselines) == 1
        baseline = baselines[0]
        assert baseline["unit_ids"] == result["unit_ids"]
        assert baseline["unit_count"] == result["unit_count"]


def test_report_exactly_covers_requested_metrics_and_slice_dimensions() -> None:
    base = _valid_document(
        "robustness_report.schema.json", "three_modes_resolved_conformal"
    )

    missing_metric = copy.deepcopy(base)
    clean = next(
        scenario
        for scenario in missing_metric["scenarios"]
        if scenario["mode"] == "clean_frozen"
    )
    clean["metrics"].remove("rmse")
    _refingerprint(clean, "scenario_fingerprint")
    _refingerprint(missing_metric, "report_fingerprint")
    with pytest.raises(ContractError, match="exactly cover"):
        validate_report(missing_metric)

    missing_slice = copy.deepcopy(base)
    clean = next(
        scenario
        for scenario in missing_slice["scenarios"]
        if scenario["mode"] == "clean_frozen"
    )
    clean["slice_by"] = sorted([*clean["slice_by"], "target"])
    _refingerprint(clean, "scenario_fingerprint")
    _refingerprint(missing_slice, "report_fingerprint")
    with pytest.raises(ContractError, match="has no exact .*target"):
        validate_report(missing_slice)


@pytest.mark.parametrize(
    ("mode", "field", "value", "expected_error"),
    [
        ("clean_frozen", "predictor_status", "refit", "clean_frozen predictor"),
        (
            "matched_recalibration",
            "calibration_status",
            "reused",
            "matched_recalibration did not create",
        ),
        (
            "structural_refit",
            "predictor_status",
            "reused",
            "structural predictor status",
        ),
    ],
)
def test_report_lifecycle_is_mode_exact(
    mode: str, field: str, value: str, expected_error: str
) -> None:
    report = _valid_document(
        "robustness_report.schema.json", "three_modes_resolved_conformal"
    )
    scenario_id = next(
        scenario["scenario_id"]
        for scenario in report["scenarios"]
        if scenario["mode"] == mode
    )
    result = next(
        result
        for result in report["results"]
        if result["scenario_id"] == scenario_id and result["severity"] > 0.0
    )
    result[field] = value
    _refingerprint(report, "report_fingerprint")
    with pytest.raises(ContractError, match=expected_error):
        validate_report(report)


def test_conformal_ci_cannot_claim_unavailable_set_size() -> None:
    report = _valid_document(
        "robustness_report.schema.json", "three_modes_resolved_conformal"
    )
    clean = next(
        scenario
        for scenario in report["scenarios"]
        if scenario["mode"] == "clean_frozen"
    )
    clean["metrics"] = sorted([*clean["metrics"], "set_size"])
    _refingerprint(clean, "scenario_fingerprint")
    result = next(
        result
        for result in report["results"]
        if result["scenario_id"] == clean["scenario_id"]
        and result["severity"] == 0.0
        and result["slice"] == {"kind": "all", "value": None}
    )
    conformal_ci = next(
        interval
        for interval in result["confidence_intervals"]
        if interval["metric_family"] == "conformal"
    )
    conformal_ci["metric"] = "set_size"
    _refingerprint(report, "report_fingerprint")
    with pytest.raises(ContractError, match="unavailable metric value"):
        validate_report(report)


def test_production_report_is_explicitly_point_only() -> None:
    fixture = load_json(FIXTURES["robustness_report.schema.json"])
    report = next(
        case["document"]
        for case in fixture["valid_cases"]
        if case["id"] == "production_point_only"
    )

    assert report["calibration_artifact_checksum"] is None
    assert report["calibration_artifacts"] == []
    assert report["conformal_prediction_blocks"] == []
    assert report["conformal_metric_sets"] == []
    assert report["provenance"]["artifact_checksums"] == []
    assert len(report["results"]) == 2
    for result in report["results"]:
        assert isinstance(result["severity"], float)
        assert result["calibration_status"] == "absent"
        assert result["coverage_guarantee_status"] == "unavailable"
        assert result["conformal_prediction_block_fingerprint"] is None
        assert result["conformal_metric_set_id"] is None
        assert all(
            interval["metric_family"] == "point"
            for interval in result["confidence_intervals"]
        )


def test_refit_and_recalibration_evidence_is_auditable() -> None:
    fixture = load_json(FIXTURES["robustness_report.schema.json"])
    report = next(
        case["document"]
        for case in fixture["valid_cases"]
        if case["id"] == "three_modes_resolved_conformal"
    )
    artifacts = {
        artifact["artifact_id"]: artifact
        for artifact in report["calibration_artifacts"]
    }
    base = artifacts["calibration:split.v1"]
    matched = artifacts["calibration:matched.v1"]
    structural = artifacts["calibration:structural.v1"]

    assert matched["predictor_binding"] == base["predictor_binding"]
    for artifact, scenario_id, severity in (
        (matched, "scenario:matched.noise", 0.5),
        (structural, "scenario:structural.node", 1.0),
    ):
        diagnostics = artifact["diagnostics"]
        assert diagnostics["scenario_id"] == scenario_id
        assert diagnostics["severity"] == severity
        assert (
            diagnostics["calibration_input_fingerprint"]
            == artifact["calibration_cohort"]["content_fingerprint"]
        )

    base_predictor = base["predictor_binding"]
    structural_predictor = structural["predictor_binding"]
    for field in (
        "campaign_fingerprint",
        "controller_fingerprint",
        "data_bindings",
        "predictor_node_ids",
        "target_processing_fingerprint",
        "training_influence_fingerprint",
    ):
        assert structural_predictor[field] == base_predictor[field]
    assert structural["training_influence"] == base["training_influence"]
    for field in (
        "plan_id",
        "graph_fingerprint",
        "selected_variant_id",
        "selected_variant_fingerprint",
        "training_outcome_fingerprint",
    ):
        assert structural_predictor[field] != base_predictor[field]
    assert structural_predictor["selected_variant_id"] == (
        f"variant:{structural_predictor['selected_variant_fingerprint'][:16]}"
    )

    structural_scenario = next(
        scenario
        for scenario in report["scenarios"]
        if scenario["mode"] == "structural_refit"
    )
    predictor_nodes = set(base_predictor["predictor_node_ids"])
    assert set(structural_scenario["node_ids"]) <= predictor_nodes


def test_standalone_joint_max_metrics_are_exact_and_unsliced() -> None:
    fixture = load_json(FIXTURES["conformal_metric_set.schema.json"])
    metric_set = next(
        case["document"]
        for case in fixture["valid_cases"]
        if case["id"] == "joint_max_exact_metrics"
    )

    assert metric_set["multi_target_policy"] == "joint_max"
    assert [record["coverage"] for record in metric_set["records"]] == [0.8, 0.9]
    assert [record["mean_width"] for record in metric_set["records"]] == [6.0, 8.0]
    assert [record["median_width"] for record in metric_set["records"]] == [6.0, 8.0]
    assert [record["interval_score"] for record in metric_set["records"]] == [6.0, 8.0]
    for record in metric_set["records"]:
        assert record["target_name"] is None
        assert record["slice"] == {"kind": "all", "value": None}
        assert record["guarantee_status"] == "joint_coverage"
        assert record["empirical_coverage"] == 1.0


def test_oracle_has_restricted_imports_and_no_production_dependency() -> None:
    source_path = ROOT / "parity" / "conformal" / "oracle.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(
                alias.name.split(".", maxsplit=1)[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", maxsplit=1)[0])
    assert imported_roots <= {
        "__future__",
        "copy",
        "decimal",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "struct",
        "typing",
        "unicodedata2",
    }


def test_fixture_generator_is_byte_identical_in_an_isolated_root(
    tmp_path: Path,
) -> None:
    fixture_rel = Path(
        "examples/fixtures/conformal/split_absolute_residual_physical_sample.v1.json"
    )
    seeded_fixture = tmp_path / fixture_rel
    seeded_fixture.parent.mkdir(parents=True)
    shutil.copy2(ROOT / fixture_rel, seeded_fixture)
    (tmp_path / "examples/fixtures/robustness").mkdir(parents=True)
    generated_rels = (
        fixture_rel,
        Path("examples/fixtures/conformal/calibration_artifacts.v1.json"),
        Path("examples/fixtures/conformal/cohort_manifest_roles.v1.json"),
        Path("examples/fixtures/conformal/conformal_prediction_blocks.v1.json"),
        Path("examples/fixtures/conformal/conformal_metric_sets.v1.json"),
        Path("examples/fixtures/conformal/domain_assessment_blocks.v1.json"),
        Path("examples/fixtures/conformal/decision_blocks.v1.json"),
        Path("examples/fixtures/robustness/robustness_scenarios.v1.json"),
        Path("examples/fixtures/robustness/robustness_reports.v1.json"),
    )

    generate(output_root=tmp_path, include_pack=False)
    first = {
        relative: (tmp_path / relative).read_bytes() for relative in generated_rels
    }
    assert first == {
        relative: (ROOT / relative).read_bytes() for relative in generated_rels
    }
    generate(output_root=tmp_path, include_pack=False)
    second = {
        relative: (tmp_path / relative).read_bytes() for relative in generated_rels
    }
    assert second == first


def test_conformance_pack_refuses_a_symlinked_path_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_docs = tmp_path / "real-docs"
    real_docs.mkdir()
    (tmp_path / "docs").symlink_to(real_docs, target_is_directory=True)
    monkeypatch.setattr(production_contracts, "ROOT", tmp_path)
    with pytest.raises(production_contracts.ContractError, match="symbolic link"):
        production_contracts.validate_conformal_robustness_pack(load_json(PACK_PATH))


def test_versioned_pack_checksums_and_tcv1_identity() -> None:
    pack = load_json(PACK_PATH)
    assert pack == conformance_pack()
    assert pack["schema_version"] == 1
    assert pack["pack_id"] == "dag-ml.conformal-robustness-conformance.v1"
    paths = [entry["path"] for entry in pack["artifacts"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    for entry in pack["artifacts"]:
        assert file_sha256(ROOT / entry["path"]) == entry["sha256"]
    assert pack["pack_checksum"] == fingerprint_without(pack, "pack_checksum")


def test_conformal_pack_hashes_transitive_schema_dependencies() -> None:
    assert not missing_schema_dependencies(ROOT, PACK_ARTIFACTS)


def test_conformal_pack_refuses_missing_true_two_hop_schema_dependency() -> None:
    seed = "docs/contracts/training_outcome.schema.json"
    intermediate = "docs/contracts/execution_bundle.schema.json"
    pack = copy.deepcopy(load_json(PACK_PATH))
    omitted = "docs/contracts/selection_decision.schema.json"
    assert seed in BASE_PACK_ARTIFACTS
    assert intermediate not in BASE_PACK_ARTIFACTS
    assert omitted not in BASE_PACK_ARTIFACTS
    intermediate_id = load_json(ROOT / intermediate)["$id"]
    omitted_id = load_json(ROOT / omitted)["$id"]
    seed_wire = json.dumps(load_json(ROOT / seed), sort_keys=True)
    intermediate_wire = json.dumps(load_json(ROOT / intermediate), sort_keys=True)
    assert intermediate_id in seed_wire
    assert omitted_id not in seed_wire
    assert omitted_id in intermediate_wire
    assert {seed, intermediate, omitted} <= set(
        schema_dependency_closure(ROOT, [seed]).paths
    )
    assert omitted in {entry["path"] for entry in pack["artifacts"]}
    pack["artifacts"] = [
        entry for entry in pack["artifacts"] if entry["path"] != omitted
    ]
    pack["pack_checksum"] = fingerprint_without(pack, "pack_checksum")
    with pytest.raises(
        production_contracts.ContractError,
        match="selection_decision.schema.json",
    ):
        production_contracts.validate_conformal_robustness_pack(pack)


def test_conformal_pack_refuses_missing_schema_dependency_resolver() -> None:
    pack = copy.deepcopy(load_json(PACK_PATH))
    omitted = "parity/schema_dependencies.py"
    assert omitted in {entry["path"] for entry in pack["artifacts"]}
    pack["artifacts"] = [
        entry for entry in pack["artifacts"] if entry["path"] != omitted
    ]
    pack["pack_checksum"] = fingerprint_without(pack, "pack_checksum")
    with pytest.raises(
        production_contracts.ContractError,
        match="schema_dependencies.py",
    ):
        production_contracts.validate_conformal_robustness_pack(pack)


def test_schema_dependency_resolver_detects_cycles_unresolved_and_escape(
    tmp_path: Path,
) -> None:
    schema_dir = tmp_path / "docs" / "contracts"
    schema_dir.mkdir(parents=True)
    first = schema_dir / "first.schema.json"
    second = schema_dir / "second.schema.json"

    def write(path: Path, schema_id: str, ref: str) -> None:
        path.write_text(
            json.dumps(
                {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$id": schema_id,
                    "$ref": ref,
                }
            ),
            encoding="utf-8",
        )

    write(first, "https://example.test/first", "second.schema.json")
    write(second, "https://example.test/second", "first.schema.json")
    closure = schema_dependency_closure(
        tmp_path, ["docs/contracts/first.schema.json"]
    )
    assert closure.cycles
    assert set(closure.paths) == {
        "docs/contracts/first.schema.json",
        "docs/contracts/second.schema.json",
    }

    write(second, "https://example.test/second", "https://example.test/missing")
    with pytest.raises(SchemaDependencyError, match="unresolved"):
        schema_dependency_closure(tmp_path, ["docs/contracts/first.schema.json"])

    write(second, "https://example.test/second", "../../../outside.schema.json")
    with pytest.raises(SchemaDependencyError, match="escapes"):
        schema_dependency_closure(tmp_path, ["docs/contracts/first.schema.json"])

    write(second, "https://example.test/second", "first.schema.json")
    real_dir = tmp_path / "real-schemas"
    real_dir.mkdir()
    linked_target = real_dir / "linked.schema.json"
    write(linked_target, "https://example.test/linked", "#")
    linked = schema_dir / "linked.schema.json"
    linked.symlink_to(linked_target)
    with pytest.raises(SchemaDependencyError, match="symbolic link"):
        schema_dependency_closure(tmp_path, ["docs/contracts/first.schema.json"])
    linked.unlink()

    duplicate = schema_dir / "duplicate.schema.json"
    write(duplicate, "https://example.test/first", "#")
    with pytest.raises(SchemaDependencyError, match=r"\$id .* duplicated"):
        schema_dependency_closure(tmp_path, ["docs/contracts/first.schema.json"])

    linked_root = tmp_path / "linked-schema-directory"
    (linked_root / "docs").mkdir(parents=True)
    real_contracts = linked_root / "real-contracts"
    real_contracts.mkdir()
    write(
        real_contracts / "root.schema.json",
        "https://example.test/root",
        "#",
    )
    (linked_root / "docs" / "contracts").symlink_to(
        real_contracts, target_is_directory=True
    )
    with pytest.raises(SchemaDependencyError, match="symbolic link"):
        schema_dependency_closure(
            linked_root, ["docs/contracts/root.schema.json"]
        )
