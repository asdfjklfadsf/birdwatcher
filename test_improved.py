import os
import sys
import types
import unittest
from datetime import timedelta
from unittest.mock import patch

# Keep logic tests lightweight; ML packages are loaded only when real models start.
for name in ("cv2", "torch", "transformers", "ultralytics"):
    sys.modules.setdefault(name, types.ModuleType(name))

from birdwatcher.classification import candidate_species_names, encode_accepted_crops
from birdwatcher.config import load_runtime_config
from birdwatcher.tracking import (
    ActiveEventTracker,
    TrackedDetection,
    box_iou,
    center_distance_ratio,
    choose_initial_detection,
    collect_tracked_crops,
    match_tracked_detection,
)


class TrackingAndEventTests(unittest.TestCase):
    def test_box_iou_detects_overlap(self):
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_initial_detection_prefers_confidence_weighted_area(self):
        small = TrackedDetection("small", 0.95, (0, 0, 20, 20))
        large = TrackedDetection("large", 0.80, (0, 0, 40, 40))
        self.assertIs(choose_initial_detection([small, large]), large)

    def test_tracking_rejects_unrelated_distant_bird(self):
        current = (100, 100, 180, 180)
        distant = TrackedDetection("other", 0.99, (600, 500, 700, 600))
        self.assertIsNone(match_tracked_detection(current, [distant]))

    def test_motion_prediction_avoids_switching_to_crossing_bird(self):
        previous = (0, 0, 10, 10)
        current = (10, 0, 20, 10)
        continued = TrackedDetection("continued", 0.70, (20, 0, 30, 10))
        crossing = TrackedDetection("crossing", 0.99, (10, 0, 20, 10))
        matched = match_tracked_detection(current, [crossing, continued], previous)
        self.assertIs(matched, continued)

    def test_active_event_suppresses_same_bird_but_not_different_bird(self):
        tracker = ActiveEventTracker(clear_seconds=3.0, max_age_seconds=600.0)
        tracker.mark_event((100, 100, 180, 180), now=0.0)
        same = TrackedDetection("same", 0.8, (105, 102, 185, 182))
        different = TrackedDetection("different", 0.8, (500, 100, 580, 180))
        new = tracker.partition_new_detections([same, different], now=1.0)
        self.assertEqual(new, [different])
        self.assertEqual(tracker.active_count, 1)

    def test_active_event_closes_after_absence(self):
        tracker = ActiveEventTracker(clear_seconds=3.0, max_age_seconds=600.0)
        box = (100, 100, 180, 180)
        tracker.mark_event(box, now=0.0)
        tracker.observe_no_detection(now=4.0)
        self.assertEqual(tracker.active_count, 0)
        detection = TrackedDetection("bird", 0.8, box)
        self.assertEqual(tracker.partition_new_detections([detection], now=4.0), [detection])

    def test_collect_tracked_crops_follows_one_track(self):
        class Capture:
            def __init__(self):
                self.frames = iter(["frame2", "frame3"])

            def read(self):
                return True, next(self.frames)

        initial = TrackedDetection("crop1", 0.8, (0, 0, 10, 10))
        detections = {
            "frame2": [TrackedDetection("crop2", 0.8, (10, 0, 20, 10))],
            "frame3": [
                TrackedDetection("other", 0.99, (10, 0, 20, 10)),
                TrackedDetection("crop3", 0.8, (20, 0, 30, 10)),
            ],
        }
        settings = types.SimpleNamespace(burst_frames=3, burst_frame_interval=0.0)
        result = collect_tracked_crops(
            Capture(),
            object(),
            initial,
            settings,
            detect_fn=lambda _models, frame, _settings: detections[frame],
            sleep_fn=lambda _seconds: None,
            clock_fn=lambda: 0.0,
        )
        self.assertEqual([item.crop for item in result], ["crop1", "crop2", "crop3"])
        self.assertLess(center_distance_ratio(result[1].box, result[2].box), 1.25)


class ClassificationPipelineTests(unittest.TestCase):
    def test_broad_model_candidates_are_added_to_bioclip_candidates(self):
        names = candidate_species_names(
            {"House Finch", "Northern Cardinal"},
            [("Osprey", 0.70), ("House Finch", 0.20)],
        )
        self.assertIn("Osprey", names)
        self.assertIn("Northern Cardinal", names)

    def test_each_crop_is_encoded_once_and_embedding_is_reused(self):
        detections = [
            TrackedDetection("a", 0.8, (0, 0, 10, 10)),
            TrackedDetection("b", 0.8, (20, 0, 30, 10)),
        ]
        encoded_calls = []
        scored_embeddings = []

        def encoder(image):
            encoded_calls.append(image)
            return f"embedding-{image}"

        def scorer(embedding):
            scored_embeddings.append(embedding)
            return 0.9 if embedding.endswith("a") else 0.4

        accepted = encode_accepted_crops(
            detections,
            models=None,
            min_presence_score=0.5,
            encoder=encoder,
            presence_scorer=scorer,
        )
        self.assertEqual(encoded_calls, ["a", "b"])
        self.assertEqual(scored_embeddings, ["embedding-a", "embedding-b"])
        self.assertEqual([item.detection.crop for item in accepted], ["a"])


class ConfigurationAndEntrypointTests(unittest.TestCase):
    def test_dotenv_style_override_controls_event_confidence(self):
        env = {
            "SMTP_HOST": "smtp.example.com",
            "EMAIL_FROM": "from@example.com",
            "EMAIL_TO": "to@example.com",
            "MIN_EVENT_DETECTOR_CONFIDENCE": "0.15",
            "DETECTION_FLOOR_CONFIDENCE": "0.05",
            "EVENT_CLEAR_SECONDS": "4.5",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "birdwatcher.config.load_dotenv", return_value=None
        ):
            runtime = load_runtime_config()
        self.assertAlmostEqual(runtime.settings.min_event_detector_confidence, 0.15)
        self.assertAlmostEqual(runtime.event_clear_seconds, 4.5)
        self.assertEqual(runtime.active_event_max_age, timedelta(minutes=10))

    def test_main_is_direct_modular_entrypoint(self):
        import main
        from birdwatcher import app

        self.assertIs(main.main, app.main)
        self.assertEqual(main.__name__, "main")


if __name__ == "__main__":
    unittest.main()
