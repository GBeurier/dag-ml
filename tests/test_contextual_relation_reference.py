"""Both independent semantic references consume a genuine new V2 owner fixture."""
import copy
import json
import unittest
from pathlib import Path
from scripts import validate_training_replay_contracts as reference
from parity.training import training_replay_oracle as independent

ROOT = Path(__file__).resolve().parents[1]
FX = ROOT / "crates/dag-ml-core/tests/fixtures/contextual_replay_v2"

class ContextualRelations(unittest.TestCase):
    def test_v2_chosen_relations_and_binding_membership(self):
        package = json.loads((FX / "package.json").read_text())
        replay = json.loads((FX / "predict-replay.json").read_text())
        envelopes = json.loads((FX / "predict-envelopes.json").read_text())
        for function in [reference.validate_replay_output_cohort, independent._validate_output_cohort]:
            fixture = {"envelopes": copy.deepcopy(envelopes)}
            # Valid development records cannot enlarge V2 current membership.
            for envelope in fixture["envelopes"].values():
                envelope["coordinator_relations"]["records"][0]["sample_id"] = "sample:development"
            function(replay["outputs"], fixture, "synthetic V2", package["effective_plan"])
            alien = copy.deepcopy(replay["outputs"])
            alien[0]["predictions"][0]["sample_ids"][0] = "sample:development"
            with self.assertRaisesRegex((reference.ContractError, independent.ContractError), "outside current cohort"):
                function(alien, fixture, "synthetic alien", package["effective_plan"])

if __name__ == "__main__":
    unittest.main()
