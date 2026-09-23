"""Cross-language selection through the Python facade."""

import dag_ml


def test_select_candidate_ranks_branch_scores_in_native_core() -> None:
    policy = {
        "id": "select:branch",
        "metric": {"name": "rmse", "objective": "minimize"},
    }
    candidates = [
        {"candidate_id": "branch:pls", "metrics": {"rmse": 2.0}},
        {"candidate_id": "branch:ridge", "metrics": {"rmse": 1.0}},
    ]
    decision = dag_ml.select_candidate(policy, candidates)
    assert decision["selected_candidate_id"] == "branch:ridge"
    assert [candidate["candidate_id"] for candidate in decision["ranked_candidates"]] == [
        "branch:ridge", "branch:pls",
    ]
