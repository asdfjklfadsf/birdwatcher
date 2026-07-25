"""Regressions for the detection-sweep, cache, and alias-identity hardening."""
import sys
import types
import unittest

for name in ("cv2", "torch", "transformers", "ultralytics"):
    sys.modules.setdefault(name, types.ModuleType(name))

from birdwatcher import classification, region, species
from birdwatcher.constants import TEXT_FEATURE_CACHE_SIZE
from birdwatcher.region import (
    identification_plausible_species,
    regional_species,
    resolve_identification,
)
from birdwatcher.species import canonical_species_key, normalize_species_key
from birdwatcher.tracking import (
    TileSweepThrottle,
    TrackedDetection,
    deduplicate_boxes,
    detect_birds,
)


class _SweepModels:
    """Counts detector passes and replays canned boxes per image size."""

    def __init__(self, boxes_by_size=None):
        self._boxes_by_size = boxes_by_size or {}
        self.calls = []

    def collect_bird_boxes(self, frame, imgsz, confidence):
        self.calls.append(imgsz)
        return list(self._boxes_by_size.get(imgsz, []))


def _settings():
    return types.SimpleNamespace(
        detection_confidence=0.35,
        detection_floor_confidence=0.05,
        max_bird_crop_aspect_ratio=2.5,
        crop_padding=0.20,
    )


class _Frame:
    """Minimal ndarray stand-in: shape plus slicing that stays nonempty."""

    shape = (900, 1200, 3)
    size = 900 * 1200 * 3

    def __getitem__(self, _item):
        return self

    def copy(self):
        return self


class SpeciesIdentityTests(unittest.TestCase):
    def test_plausible_species_set_is_keyed_canonically(self):
        plausible = identification_plausible_species("northern_nj", 7)
        self.assertIn(canonical_species_key("Tit Mouse"), plausible)
        self.assertNotIn(normalize_species_key("Tit Mouse"), plausible)

    def test_every_alias_of_a_regional_species_stays_plausible(self):
        """A canonicalized winner must never be rejected as out of region."""
        plausible = identification_plausible_species("northern_nj", 7)
        regional = regional_species("northern_nj", 7)
        for source, canonical in species._SPECIES_ALIASES.items():
            if canonical_species_key(source) in regional:
                self.assertIn(canonical_species_key(canonical), plausible)

    def test_canonical_winner_is_not_marked_out_of_region(self):
        result = resolve_identification(
            [[("Tufted Titmouse", 0.90), ("House Finch", 0.10)]] * 4,
            min_votes=4,
            min_confidence=0.60,
            min_margin=0.20,
            plausible_species=identification_plausible_species("northern_nj", 7),
        )
        self.assertFalse(result.uncertain)
        self.assertEqual(result.display_name, "Tufted Titmouse")

    def test_blending_is_defined_once_and_collapses_aliases(self):
        self.assertIs(
            classification.combine_classifier_predictions,
            region.combine_classifier_predictions,
        )
        combined = dict(
            region.combine_classifier_predictions(
                [("Tufted Titmouse", 1.0)], [("Tit Mouse", 1.0)], 0.65
            )
        )
        self.assertEqual(list(combined), ["Tufted Titmouse"])


class DetectionSweepTests(unittest.TestCase):
    def test_overlapping_duplicates_collapse_to_the_best_box(self):
        kept = deduplicate_boxes(
            [
                (100, 100, 180, 180, 0.40),
                (104, 102, 184, 182, 0.90),
                (600, 100, 680, 180, 0.50),
            ]
        )
        self.assertEqual(
            sorted(kept), sorted([(104, 102, 184, 182, 0.90), (600, 100, 680, 180, 0.50)])
        )

    def test_distinct_adjacent_birds_are_not_merged(self):
        kept = deduplicate_boxes(
            [(100, 100, 180, 180, 0.90), (170, 100, 250, 180, 0.85)]
        )
        self.assertEqual(len(kept), 2)

    def test_suppression_boundary_is_the_configured_threshold(self):
        below = deduplicate_boxes([(100, 100, 180, 180, 0.90), (130, 100, 210, 180, 0.85)])
        above = deduplicate_boxes([(100, 100, 180, 180, 0.90), (124, 100, 204, 180, 0.85)])
        self.assertEqual(len(below), 2, "IoU 0.45 must survive")
        self.assertEqual(len(above), 1, "IoU 0.54 must collapse")

    def test_primary_pass_overlaps_are_not_suppressed(self):
        """Two birds the detector kept must not be merged by a second NMS pass."""
        models = _SweepModels(
            {640: [(100, 100, 180, 180, 0.90), (120, 100, 200, 180, 0.85)]}
        )
        detections = detect_birds(models, _Frame(), _settings())
        self.assertEqual(models.calls, [640])
        self.assertEqual(len(detections), 2)

    def test_floor_pass_overlaps_are_not_suppressed(self):
        models = _SweepModels(
            {1280: [(100, 100, 180, 180, 0.09), (120, 100, 200, 180, 0.08)]}
        )
        detections = detect_birds(models, _Frame(), _settings())
        self.assertEqual(models.calls, [640, 1280])
        self.assertEqual(len(detections), 2)

    def test_tile_sweep_is_skipped_when_disallowed(self):
        models = _SweepModels()
        detect_birds(models, _Frame(), _settings(), allow_tile_sweep=False)
        self.assertEqual(models.calls, [640, 1280])

    def test_tile_sweep_runs_when_allowed_and_nothing_was_found(self):
        models = _SweepModels()
        detect_birds(models, _Frame(), _settings(), allow_tile_sweep=True)
        self.assertEqual(models.calls, [640, 1280] + [640] * 9)

    def test_no_fallback_passes_when_the_primary_sweep_finds_a_bird(self):
        models = _SweepModels({640: [(100, 100, 180, 180, 0.80)]})
        detections = detect_birds(models, _Frame(), _settings())
        self.assertEqual(models.calls, [640])
        self.assertEqual(len(detections), 1)


class TileSweepThrottleTests(unittest.TestCase):
    def test_throttle_allows_one_sweep_per_interval(self):
        throttle = TileSweepThrottle(5.0)
        self.assertTrue(throttle.allow(100.0))
        self.assertFalse(throttle.allow(101.0))
        self.assertFalse(throttle.allow(104.9))
        self.assertTrue(throttle.allow(105.0))

    def test_zero_interval_never_blocks(self):
        throttle = TileSweepThrottle(0.0)
        self.assertTrue(all(throttle.allow(float(tick)) for tick in range(5)))

    def test_throttle_is_not_spent_when_the_primary_pass_finds_a_bird(self):
        """Budget must survive frames that never needed a sweep."""
        throttle = TileSweepThrottle(5.0)

        found = _SweepModels({640: [(100, 100, 180, 180, 0.80)]})
        detect_birds(found, _Frame(), _settings(), tile_sweep=throttle, now=100.0)
        self.assertEqual(found.calls, [640])

        # One second later the bird is lost. The sweep must still be affordable.
        lost = _SweepModels()
        detect_birds(lost, _Frame(), _settings(), tile_sweep=throttle, now=101.0)
        self.assertEqual(lost.calls, [640, 1280] + [640] * 9)

    def test_throttle_is_spent_only_by_frames_that_actually_sweep(self):
        throttle = TileSweepThrottle(5.0)

        first = _SweepModels()
        detect_birds(first, _Frame(), _settings(), tile_sweep=throttle, now=100.0)
        self.assertEqual(len(first.calls), 11)

        within_interval = _SweepModels()
        detect_birds(within_interval, _Frame(), _settings(), tile_sweep=throttle, now=102.0)
        self.assertEqual(within_interval.calls, [640, 1280])

        after_interval = _SweepModels()
        detect_birds(after_interval, _Frame(), _settings(), tile_sweep=throttle, now=105.0)
        self.assertEqual(len(after_interval.calls), 11)

    def test_tracked_burst_shares_the_throttle_instead_of_sweeping_every_frame(self):
        """A nine-frame burst used to allow eight unthrottled tile sweeps."""
        from birdwatcher.tracking import collect_tracked_crops

        swept = []
        now = [0.0]
        throttle = TileSweepThrottle(5.0)

        class Capture:
            def read(self):
                return True, "frame"

        def fake_detect(_models, _frame, _burst_settings, *, tile_sweep=None, now=None):
            models = _SweepModels()
            detect_birds(models, _Frame(), _settings(), tile_sweep=tile_sweep, now=now)
            swept.append(len(models.calls) == 11)
            return []

        collect_tracked_crops(
            Capture(),
            object(),
            TrackedDetection("crop", 0.8, (0, 0, 10, 10)),
            types.SimpleNamespace(burst_frames=9, burst_frame_interval=1.0),
            detect_fn=fake_detect,
            sleep_fn=lambda seconds: now.__setitem__(0, now[0] + max(0.0, seconds)),
            clock_fn=lambda: now[0],
            tile_sweep=throttle,
        )

        self.assertEqual(len(swept), 8)
        self.assertLessEqual(sum(swept), 3)

    def test_burst_sweeps_freely_when_no_throttle_is_supplied(self):
        from birdwatcher.tracking import collect_tracked_crops

        received = []

        class Capture:
            def read(self):
                return True, "frame"

        def fake_detect(_models, _frame, _settings, *, tile_sweep=None, now=None):
            received.append(tile_sweep)
            return []

        collect_tracked_crops(
            Capture(),
            object(),
            TrackedDetection("crop", 0.8, (0, 0, 10, 10)),
            types.SimpleNamespace(burst_frames=4, burst_frame_interval=0.0),
            detect_fn=fake_detect,
            sleep_fn=lambda _seconds: None,
            clock_fn=lambda: 0.0,
        )

        self.assertEqual(received, [None, None, None])

    def test_run_loop_rate_limits_the_tile_sweep(self):
        from datetime import timedelta
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from birdwatcher import app

        requested = []
        clock = iter([100.0, 101.0, 102.0, 106.0, 107.0])

        class Capture:
            def read(self):
                return True, _Frame()

            def release(self):
                return None

        def fake_sleep(_seconds):
            if len(requested) >= 5:
                raise KeyboardInterrupt

        with TemporaryDirectory() as temp:
            settings = types.SimpleNamespace(
                image_dir=Path(temp),
                camera=0,
                camera_width=1280,
                camera_height=960,
                camera_fps=5.0,
                scan_interval=0.0,
                tile_sweep_interval=5.0,
                detector_model="d",
                detector_sha256="x",
                classifier_model="c",
                classifier_revision="r",
                local_classifier_model="l",
                local_classifier_revision="lr",
                email=object(),
            )
            config = types.SimpleNamespace(
                settings=settings,
                event_clear_seconds=6.0,
                active_event_max_age=timedelta(minutes=10),
            )

            def fake_detect(_models, _frame, _scan_settings, *, tile_sweep=None, now=None):
                models = _SweepModels()
                detect_birds(models, _Frame(), _settings(), tile_sweep=tile_sweep, now=now)
                requested.append(len(models.calls) == 11)
                return []

            with (
                patch("birdwatcher.app.BirdModels", return_value=object()),
                patch("birdwatcher.app.open_camera", return_value=Capture()),
                patch("birdwatcher.app.detect_birds", side_effect=fake_detect),
                patch("birdwatcher.app.EmailRetryQueue"),
                patch("birdwatcher.app.time.monotonic", side_effect=lambda: next(clock)),
                patch("birdwatcher.app.time.sleep", side_effect=fake_sleep),
            ):
                app.run(config)

        # Scans at t=100, 101, 102, 106, 107 with a 5s interval: only the first
        # and the one at/after t=105 may pay for the nine-tile sweep.
        self.assertEqual(requested, [True, False, False, True, False])


class TextFeatureCacheTests(unittest.TestCase):
    def test_cache_is_bounded_and_keeps_recent_entries(self):
        stored = []

        class FakeTorch:
            @staticmethod
            def inference_mode():
                class Ctx:
                    def __enter__(self):
                        return None

                    def __exit__(self, *_args):
                        return False

                return Ctx()

        class FakeFeatures:
            def __truediv__(self, _other):
                return self

            def norm(self, **_kwargs):
                return self

            def reshape(self, *_args):
                return self

            def mean(self, **_kwargs):
                return self

            @property
            def T(self):
                return self

            def contiguous(self):
                return self

        class FakeClassifier:
            @staticmethod
            def encode_text(_prompts):
                stored.append(1)
                return FakeFeatures()

        models = types.SimpleNamespace(
            torch=FakeTorch(),
            device="cpu",
            local_classifier=FakeClassifier(),
            local_tokenizer=lambda prompts: types.SimpleNamespace(to=lambda _device: prompts),
        )

        for index in range(TEXT_FEATURE_CACHE_SIZE + 5):
            classification._species_text_features(models, (f"Species {index}",))

        self.assertEqual(len(models._runtime_text_cache), TEXT_FEATURE_CACHE_SIZE)
        newest = (f"Species {TEXT_FEATURE_CACHE_SIZE + 4}",)
        self.assertIn(newest, models._runtime_text_cache)
        self.assertNotIn(("Species 0",), models._runtime_text_cache)


if __name__ == "__main__":
    unittest.main()
