import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

# Core logic can be tested without loading the real ML packages.
for name in ("cv2", "torch", "transformers", "ultralytics"):
    sys.modules.setdefault(name, types.ModuleType(name))

import legacy_main as core


class FakeCV2:
    @staticmethod
    def imwrite(path, image):
        Path(path).write_bytes(b"jpeg")
        return True


class CoreLogicTests(unittest.TestCase):
    def test_season_is_derived_from_observation_month(self):
        self.assertEqual(core.season_for_month(1), "winter")
        self.assertEqual(core.season_for_month(4), "spring")
        self.assertEqual(core.season_for_month(7), "summer")
        self.assertEqual(core.season_for_month(10), "fall")

    def test_regional_prior_reranks_plausible_species(self):
        plausible = core.regional_species("northern_nj", 7)
        preferred = core.preferred_regional_species("northern_nj", 7)
        adjusted = core.apply_regional_prior(
            [("Azaras Spinetail", 0.55), ("Northern Cardinal", 0.30), ("House Finch", 0.15)],
            preferred,
            weight=3.0,
            plausible_species=plausible,
        )
        self.assertEqual(adjusted[0][0], "Northern Cardinal")
        self.assertAlmostEqual(sum(score for _, score in adjusted), 1.0)

    def test_hybrid_combination_preserves_unusual_candidate(self):
        combined = core.combine_classifier_predictions(
            local_predictions=[("House Finch", 0.80), ("House Sparrow", 0.20)],
            global_predictions=[("African Firefinch", 0.70), ("House Finch", 0.30)],
            local_weight=0.65,
        )
        labels = [name for name, _ in combined]
        self.assertEqual(combined[0][0], "House Finch")
        self.assertIn("African Firefinch", labels)
        self.assertAlmostEqual(sum(score for _, score in combined), 1.0)

    def test_event_gate_rejects_single_frame_false_positive(self):
        class Crop:
            shape = (120, 90, 3)

        accepted, reason = core.validate_bird_event(
            [(Crop(), 0.35)],
            min_frames=4,
            min_median_confidence=0.07,
            max_aspect_ratio=2.5,
        )
        self.assertEqual(accepted, [])
        self.assertIn("1 valid detection", reason)

    def test_event_gate_accepts_repeated_low_confidence_small_bird_detections(self):
        class Crop:
            shape = (120, 90, 3)

        detections = [(Crop(), score) for score in (0.06, 0.07, 0.08, 0.09)]
        accepted, reason = core.validate_bird_event(
            detections,
            min_frames=4,
            min_median_confidence=0.07,
            max_aspect_ratio=2.5,
        )
        self.assertEqual(accepted, detections)
        self.assertIsNone(reason)

    def test_presence_gate_rejects_empty_feeder_crops(self):
        crops = [("dish-a", 0.69), ("dish-b", 0.66), ("dish-c", 0.71)]
        scores = {"dish-a": 0.01, "dish-b": 0.05, "dish-c": 0.03}
        accepted, reason, measured = core.validate_visual_bird_presence(
            crops,
            scores.__getitem__,
            min_score=0.50,
            min_frames=2,
        )
        self.assertEqual(accepted, [])
        self.assertIsNotNone(reason)
        self.assertEqual(measured, [0.01, 0.05, 0.03])

    def test_consensus_accepts_strong_multi_frame_identification(self):
        result = core.resolve_identification(
            [
                [("Northern Cardinal", 0.92), ("House Finch", 0.05)],
                [("Northern Cardinal", 0.86), ("House Finch", 0.08)],
                [("Northern Cardinal", 0.82), ("House Finch", 0.10)],
                [("Northern Cardinal", 0.80), ("House Finch", 0.12)],
            ],
            min_votes=4,
            min_confidence=0.60,
            min_margin=0.20,
            plausible_species=core.identification_plausible_species("northern_nj", 7),
        )
        self.assertFalse(result.uncertain)
        self.assertEqual(result.display_name, "Northern Cardinal")

    def test_consensus_marks_out_of_region_winner_uncertain(self):
        result = core.resolve_identification(
            [[("Azaras Spinetail", 0.97), ("Northern Cardinal", 0.02)]] * 4,
            min_votes=4,
            min_confidence=0.60,
            min_margin=0.20,
            plausible_species=core.identification_plausible_species("northern_nj", 7),
        )
        self.assertTrue(result.uncertain)
        self.assertEqual(result.display_name, "Uncertain bird")

    def test_small_saved_crop_is_upscaled(self):
        resized_shapes = []

        class FakeCV2WithResize:
            INTER_LANCZOS4 = "lanczos"

            @staticmethod
            def imwrite(path, image):
                Path(path).write_bytes(b"jpeg")
                return True

            @staticmethod
            def resize(image, size, interpolation=None):
                resized_shapes.append(tuple(size))
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        small = np.zeros((77, 112, 3), dtype=np.uint8)
        with TemporaryDirectory() as temp, patch("legacy_main.cv2", FakeCV2WithResize):
            path = core.save_bird_image(
                small,
                Path(temp),
                "test_species",
                datetime(2026, 7, 22, 14, 0, 0),
            )
        self.assertTrue(resized_shapes)
        self.assertTrue(path.name.endswith("test_species.jpg"))
        self.assertGreaterEqual(min(resized_shapes[0]), 320)

    def test_species_name_is_human_readable(self):
        self.assertEqual(
            core.format_species_name("BLACK-CAPPED CHICKADEE"),
            "Black-Capped Chickadee",
        )


if __name__ == "__main__":
    unittest.main()
