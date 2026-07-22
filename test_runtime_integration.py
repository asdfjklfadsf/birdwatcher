import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

for name in ("cv2", "torch", "transformers", "ultralytics"):
    sys.modules.setdefault(name, types.ModuleType(name))

from birdwatcher.app import process_new_event
from birdwatcher.classification import EncodedCrop
from birdwatcher.domain import IdentificationResult
from birdwatcher.tracking import ActiveEventTracker, TrackedDetection


class RuntimeIntegrationTests(unittest.TestCase):
    def _settings(self, image_dir):
        return types.SimpleNamespace(
            min_valid_bird_frames=4,
            min_event_detector_confidence=0.07,
            max_bird_crop_aspect_ratio=2.5,
            min_bird_presence_score=0.50,
            min_bird_presence_frames=2,
            sharpest_frames=4,
            region_profile="northern_nj",
            consensus_min_votes=4,
            species_min_confidence=0.60,
            species_min_margin=0.20,
            local_classifier_weight=0.65,
            regional_prior_weight=3.0,
            image_dir=Path(image_dir),
            email=object(),
        )

    def _identification(self):
        return IdentificationResult(
            display_name="Northern Cardinal",
            candidate_name="Northern Cardinal",
            confidence=0.90,
            margin=0.70,
            votes=4,
            frame_count=4,
            uncertain=False,
            top_candidates=(("Northern Cardinal", 0.90),),
        )

    def test_confirmed_event_refreshes_after_expensive_processing(self):
        with TemporaryDirectory() as temp:
            settings = self._settings(temp)
            detections = [
                TrackedDetection(object(), 0.10, (100 + i * 5, 100, 180 + i * 5, 180))
                for i in range(4)
            ]
            encoded = [EncodedCrop(item, f"embedding-{i}", 0.9) for i, item in enumerate(detections)]
            active = ActiveEventTracker(clear_seconds=3.0, max_age_seconds=600.0)
            clock_values = iter([100.0, 120.0])

            with (
                patch("birdwatcher.app.collect_tracked_crops", return_value=detections),
                patch(
                    "birdwatcher.app.validate_bird_event",
                    return_value=([(item.crop, item.score) for item in detections], None),
                ),
                patch("birdwatcher.app.encode_accepted_crops", return_value=encoded),
                patch("birdwatcher.app.image_sharpness", return_value=100.0),
                patch("birdwatcher.app.preferred_regional_species_names", return_value={"Northern Cardinal"}),
                patch("birdwatcher.app.preferred_regional_species", return_value={"northerncardinal"}),
                patch("birdwatcher.app.regional_species", return_value={"northerncardinal"}),
                patch("birdwatcher.app.identification_plausible_species", return_value={"northerncardinal"}),
                patch("birdwatcher.app.hybrid_predictions", return_value=[("Northern Cardinal", 1.0)]),
                patch("birdwatcher.app.resolve_identification", return_value=self._identification()),
                patch("birdwatcher.app.save_bird_image", return_value=Path(temp) / "bird.jpg"),
                patch("birdwatcher.app.send_email"),
            ):
                processed = process_new_event(
                    capture=object(),
                    models=object(),
                    initial=detections[0],
                    settings=settings,
                    active_events=active,
                    event_time=datetime(2026, 7, 22, 12, 0, 0),
                    clock_fn=lambda: next(clock_values),
                )

            self.assertTrue(processed)
            repeat = TrackedDetection(object(), 0.9, (118, 102, 198, 182))
            self.assertEqual(active.partition_new_detections([repeat], now=121.0), [])

    def test_event_is_refreshed_even_when_classification_raises(self):
        settings = self._settings(".")
        detections = [
            TrackedDetection(types.SimpleNamespace(shape=(80, 80, 3)), 0.10, (100 + i * 5, 100, 180 + i * 5, 180))
            for i in range(4)
        ]
        encoded = [EncodedCrop(item, f"embedding-{i}", 0.9) for i, item in enumerate(detections)]
        active = ActiveEventTracker(clear_seconds=3.0, max_age_seconds=600.0)
        clock_values = iter([100.0, 120.0])
        with (
            patch("birdwatcher.app.collect_tracked_crops", return_value=detections),
            patch("birdwatcher.app.validate_bird_event", return_value=([(item.crop, item.score) for item in detections], None)),
            patch("birdwatcher.app.encode_accepted_crops", return_value=encoded),
            patch("birdwatcher.app.image_sharpness", side_effect=RuntimeError("boom")),
        ):
            with self.assertRaises(RuntimeError):
                process_new_event(
                    capture=object(),
                    models=object(),
                    initial=detections[0],
                    settings=settings,
                    active_events=active,
                    event_time=datetime(2026, 7, 22, 12, 0, 0),
                    clock_fn=lambda: next(clock_values),
                )
        repeat = TrackedDetection(object(), 0.9, detections[-1].box)
        self.assertEqual(active.partition_new_detections([repeat], now=121.0), [])

    def test_rejected_false_event_does_not_enter_active_event_tracker(self):
        settings = self._settings(".")
        initial = TrackedDetection(object(), 0.1, (100, 100, 180, 180))
        active = ActiveEventTracker(clear_seconds=3.0, max_age_seconds=600.0)
        with (
            patch("birdwatcher.app.collect_tracked_crops", return_value=[initial]),
            patch("birdwatcher.app.validate_bird_event", return_value=([], "only one detection")),
        ):
            processed = process_new_event(
                capture=object(),
                models=object(),
                initial=initial,
                settings=settings,
                active_events=active,
                event_time=datetime(2026, 7, 22, 12, 0, 0),
                clock_fn=lambda: 100.0,
            )
        self.assertFalse(processed)
        self.assertEqual(active.active_count, 0)


if __name__ == "__main__":
    unittest.main()
