"""Native identifier bounds must preserve scope and cross-host identities."""

import pytest

from dag_ml.multimodal_methods import _bounded_identifier


@pytest.mark.parametrize("prefix", ["lineage:methods-multimodal", "artifact:methods.multimodal"])
def test_identifier_preserves_the_exact_native_boundary(prefix: str) -> None:
    coordinate = "r" * (127 - len(prefix))
    original = f"{prefix}:{coordinate}"
    assert len(original.encode("utf-8")) == 128
    assert _bounded_identifier(prefix, coordinate) == original
    bounded = _bounded_identifier(prefix, coordinate + "r")
    assert len(bounded.encode("utf-8")) <= 128
    assert bounded != original


def test_long_lineage_matches_shared_sha256_vector_and_keeps_coordinate_boundaries() -> None:
    prefix, run = "lineage:methods-multimodal", "run:" + "r" * 100
    first = _bounded_identifier(prefix, run, "model:a", "PREDICT", "base", "full")
    second = _bounded_identifier(prefix, run + ":model", "a", "PREDICT", "base", "full")
    # Both old concatenations are identical; ordered-array hashing retains scope.
    assert ":".join([run, "model:a"]) == ":".join([run + ":model", "a"])
    assert first == prefix + ":22407a3baa33ed60aad13b2a8bcde292d69e0e1cfd2f17c0c5e145bed261a681"
    assert second == prefix + ":566707755525c2a470aa50d6b58d7c3137a885f884d1e6b740240cbc0bb473bd"
    assert first != second


def test_long_refit_artifact_matches_shared_sha256_vector() -> None:
    actual = _bounded_identifier("artifact:methods.multimodal", "run:" + "r" * 100, "model:a", "base", "refit")
    assert actual == "artifact:methods.multimodal:4a2b4526eefcf65d51ca84386b191702af8245079ff0f47ee9b19957f1a9d96b"
