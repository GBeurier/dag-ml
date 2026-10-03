"""Native average wire shapes; archive gates attest complete signed outcomes."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from scripts.validate_archive_v2_contract import contract_schema_registry


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/GBeurier/dag-ml/schemas/"
AVERAGE_FIELDS = ("oof_averages", "ensemble_averages")


class TrainingOutcomeV2AveragesSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas, cls.registry = contract_schema_registry(ROOT)
        cls.schema = cls.schemas[BASE + "training_outcome.v2.schema.json"]
        fixtures = ROOT / "examples/fixtures/training/replay"
        cls.outcome = json.loads(
            (fixtures / "training_outcome_port_explicit.v2.json").read_text()
        )
        cls.native_blocks = json.loads(
            (fixtures / "native_training_averages.v2.json").read_text()
        )

    def test_native_average_fields_are_optional_and_accepted_by_the_v2_root(self) -> None:
        validator = Draft202012Validator(self.schema, registry=self.registry)
        # The historical V2 fixture has no averages. Adding exact native block
        # excerpts checks wire shape only; it does not re-sign this fixture or
        # claim the excerpt's scientific closure belongs to this other outcome.
        validator.validate(self.outcome)
        for field in AVERAGE_FIELDS:
            with self.subTest(field=field):
                self.assertNotIn(field, self.schema["required"])
                self.assertEqual(self.schema["properties"][field]["default"], [])
                self.assertNotIn(
                    field,
                    self.schemas[BASE + "training_outcome.v1.schema.json"]["properties"],
                )
                outcome = copy.deepcopy(self.outcome)
                outcome[field] = copy.deepcopy(self.native_blocks[field])
                validator.validate(outcome)
                outcome[field] = []
                validator.validate(outcome)

    def test_average_blocks_require_closed_prediction_and_target_shapes(self) -> None:
        for field in AVERAGE_FIELDS:
            validator = Draft202012Validator(
                {"$ref": BASE + "training_outcome.v2.schema.json#/properties/" + field},
                registry=self.registry,
            )
            for mutation in (
                "missing_predictions", "missing_truth", "missing_port",
                "unknown_wrapper", "unknown_prediction", "unknown_truth",
                "nonnumeric_prediction", "nonnumeric_truth", "null_array",
            ):
                with self.subTest(field=field, mutation=mutation):
                    blocks = copy.deepcopy(self.native_blocks[field])
                    block = blocks[0]
                    if mutation == "missing_predictions":
                        del block["predictions"]
                    elif mutation == "missing_truth":
                        del block["y_true"]
                    elif mutation == "missing_port":
                        del block["predictions"]["producer_port"]
                    elif mutation == "unknown_wrapper":
                        block["average_policy"] = "unchecked"
                    elif mutation == "unknown_prediction":
                        block["predictions"]["average_policy"] = "unchecked"
                    elif mutation == "unknown_truth":
                        block["y_true"]["average_policy"] = "unchecked"
                    elif mutation == "nonnumeric_prediction":
                        block["predictions"]["values"][0][0] = "1.0"
                    elif mutation == "nonnumeric_truth":
                        block["y_true"]["values"][0][0] = "1.0"
                    else:
                        blocks = None
                    with self.assertRaises(ValidationError):
                        validator.validate(blocks)

    def test_training_outcome_still_refuses_unrecognized_root_fields(self) -> None:
        outcome = copy.deepcopy(self.outcome)
        outcome["unrecognized_averages"] = self.native_blocks["oof_averages"]
        with self.assertRaises(ValidationError):
            Draft202012Validator(self.schema, registry=self.registry).validate(outcome)


if __name__ == "__main__":
    unittest.main()
