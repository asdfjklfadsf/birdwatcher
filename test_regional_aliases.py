import os
import sys
import types
import unittest
from unittest.mock import patch

for name in ("cv2", "torch", "transformers", "ultralytics"):
    sys.modules.setdefault(name, types.ModuleType(name))

from birdwatcher.classification import (
    candidate_species_names,
    combine_classifier_predictions,
)
from birdwatcher.config import load_runtime_config
from birdwatcher.constants import BROAD_REGIONAL_PRIOR_WEIGHT
from birdwatcher.species import collapse_species_aliases


class SpeciesAliasTests(unittest.TestCase):
    def test_candidate_species_names_collapse_titmouse_alias(self):
        names = candidate_species_names(
            {"Tufted Titmouse", "House Finch"},
            [("Tit Mouse", 0.70), ("House Finch", 0.20)],
        )

        self.assertIn("Tufted Titmouse", names)
        self.assertNotIn("Tit Mouse", names)
        self.assertEqual(len(names), 2)

    def test_alias_scores_are_combined_before_classifier_blending(self):
        combined = combine_classifier_predictions(
            local_predictions=[
                ("Tufted Titmouse", 0.60),
                ("House Finch", 0.40),
            ],
            global_predictions=[
                ("Tit Mouse", 0.80),
                ("House Finch", 0.20),
            ],
            local_weight=0.65,
        )
        scores = dict(combined)

        self.assertNotIn("Tit Mouse", scores)
        self.assertAlmostEqual(scores["Tufted Titmouse"], 0.67)
        self.assertAlmostEqual(scores["House Finch"], 0.33)
        self.assertAlmostEqual(sum(scores.values()), 1.0)

    def test_duplicate_aliases_within_one_prediction_list_are_collapsed(self):
        collapsed = dict(
            collapse_species_aliases(
                [
                    ("Tufted Titmouse", 0.3900),
                    ("Tit Mouse", 0.3764),
                    ("House Finch", 0.2142),
                ]
            )
        )

        self.assertAlmostEqual(collapsed["Tufted Titmouse"], 0.7664)
        self.assertNotIn("Tit Mouse", collapsed)


class RegionalWeightConfigurationTests(unittest.TestCase):
    def _load_with_weight(self, weight: str):
        env = {
            "SMTP_HOST": "smtp.example.com",
            "EMAIL_FROM": "from@example.com",
            "EMAIL_TO": "to@example.com",
            "REGIONAL_PRIOR_WEIGHT": weight,
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "birdwatcher.config.load_dotenv", return_value=None
        ):
            return load_runtime_config()

    def test_weight_below_broad_regional_weight_is_rejected_at_startup(self):
        with self.assertRaisesRegex(
            ValueError,
            r"REGIONAL_PRIOR_WEIGHT must be finite and at least 1\.5",
        ):
            self._load_with_weight("1.49")

    def test_weight_equal_to_broad_regional_weight_is_allowed(self):
        runtime = self._load_with_weight(str(BROAD_REGIONAL_PRIOR_WEIGHT))
        self.assertEqual(
            runtime.settings.regional_prior_weight,
            BROAD_REGIONAL_PRIOR_WEIGHT,
        )


if __name__ == "__main__":
    unittest.main()
