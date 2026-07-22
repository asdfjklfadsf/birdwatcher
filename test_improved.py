import sys
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

# Match the lightweight import strategy used by the existing test suite.
for name in ("cv2", "torch", "transformers", "ultralytics"):
    sys.modules.setdefault(name, types.ModuleType(name))

from birdwatcher_improved import (
    SpeciesCooldownTracker,
    TrackedDetection,
    box_iou,
    center_distance_ratio,
    choose_initial_detection,
    match_tracked_detection,
)


class ImprovedBirdWatcherTests(unittest.TestCase):
    def test_box_iou_detects_overlap(self):
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)
        self.assertEqual(box_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_initial_detection_prefers_confidence_weighted_area(self):
        small_confident = TrackedDetection("small", 0.95, (0, 0, 20, 20))
        large_clear = TrackedDetection("large", 0.80, (0, 0, 40, 40))
        self.assertIs(choose_initial_detection([small_confident, large_clear]), large_clear)

    def test_tracking_prefers_same_overlapping_bird(self):
        previous = (100, 100, 180, 180)
        same_bird = TrackedDetection("same", 0.70, (110, 105, 190, 185))
        other_bird = TrackedDetection("other", 0.95, (350, 100, 450, 200))
        matched = match_tracked_detection(previous, [other_bird, same_bird])
        self.assertIs(matched, same_bird)

    def test_tracking_rejects_unrelated_distant_bird(self):
        previous = (100, 100, 180, 180)
        distant = TrackedDetection("other", 0.99, (600, 500, 700, 600))
        self.assertIsNone(match_tracked_detection(previous, [distant]))

    def test_tracking_allows_motion_when_center_stays_plausibly_close(self):
        previous = (100, 100, 180, 180)
        moved = TrackedDetection("same", 0.70, (175, 105, 255, 185))
        self.assertLess(center_distance_ratio(previous, moved.box), 1.25)
        self.assertIs(match_tracked_detection(previous, [moved]), moved)

    def test_species_cooldown_allows_different_species(self):
        with TemporaryDirectory() as temp:
            tracker = SpeciesCooldownTracker(timedelta(minutes=10), Path(temp) / "state.json")
            now = datetime(2026, 7, 22, 12, 0, 0)
            tracker.mark_sent("Northern Cardinal", now)
            self.assertFalse(tracker.is_ready("Northern Cardinal", now + timedelta(minutes=1)))
            self.assertTrue(tracker.is_ready("Blue Jay", now + timedelta(minutes=1)))

    def test_species_cooldown_survives_restart(self):
        with TemporaryDirectory() as temp:
            state = Path(temp) / "state.json"
            now = datetime(2026, 7, 22, 12, 0, 0)
            first = SpeciesCooldownTracker(timedelta(minutes=10), state)
            first.mark_sent("Northern Cardinal", now)
            restarted = SpeciesCooldownTracker(timedelta(minutes=10), state)
            self.assertFalse(restarted.is_ready("Northern Cardinal", now + timedelta(minutes=5)))
            self.assertTrue(restarted.is_ready("Blue Jay", now + timedelta(minutes=5)))


if __name__ == "__main__":
    unittest.main()
