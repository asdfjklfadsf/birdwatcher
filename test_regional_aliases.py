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
    hybrid_predictions,
)
from birdwatcher.config import load_runtime_config
from birdwatcher.constants import BROAD_REGIONAL_PRIOR_WEIGHT
from birdwatcher.region import (
    apply_regional_prior,
    preferred_regional_species,
    regional_species,
)
from birdwatcher.species import collapse_species_aliases


class _StubModels:
    """Stands in for BirdModels: broad candidates plus BioCLIP scoring."""

    def __init__(self, raw_global):
        self._raw_global = raw_global

    def identify_species_candidates(self, bird_image, top_k=20):
        return list(self._raw_global)


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

    def test_weight_below_one_is_rejected_at_startup(self):
        with self.assertRaisesRegex(
            ValueError,
            r"REGIONAL_PRIOR_WEIGHT must be finite and at least 1",
        ):
            self._load_with_weight("0.9")

    def test_weight_equal_to_broad_regional_weight_is_allowed(self):
        runtime = self._load_with_weight(str(BROAD_REGIONAL_PRIOR_WEIGHT))
        self.assertEqual(
            runtime.settings.regional_prior_weight,
            BROAD_REGIONAL_PRIOR_WEIGHT,
        )

    def test_weight_below_broad_weight_is_accepted_and_clamped_not_crashing(self):
        """A weight under the broad multiplier used to crash every event."""
        runtime = self._load_with_weight("1.0")
        self.assertEqual(runtime.settings.regional_prior_weight, 1.0)

        with patch(
            "birdwatcher.classification.local_predictions_from_embedding",
            return_value=[("Tufted Titmouse", 0.70), ("House Finch", 0.30)],
        ):
            predictions = hybrid_predictions(
                _StubModels([("Tit Mouse", 0.60), ("House Finch", 0.40)]),
                "crop",
                "embedding",
                {"Tufted Titmouse", "House Finch"},
                preferred_regional_species("northern_nj", 7),
                regional_species("northern_nj", 7),
                runtime.settings,
            )
        self.assertAlmostEqual(sum(score for _, score in predictions), 1.0)
        self.assertEqual(predictions[0][0], "Tufted Titmouse")

    def test_neutral_weight_applies_no_regional_preference(self):
        neutral = apply_regional_prior(
            [("Azaras Spinetail", 0.55), ("Northern Cardinal", 0.45)],
            preferred_regional_species("northern_nj", 7),
            weight=1.0,
            plausible_species=regional_species("northern_nj", 7),
            plausible_weight=min(BROAD_REGIONAL_PRIOR_WEIGHT, 1.0),
        )
        self.assertEqual(neutral[0][0], "Azaras Spinetail")


if __name__ == "__main__":
    unittest.main()
