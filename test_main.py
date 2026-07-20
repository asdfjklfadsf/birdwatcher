import sys
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

# Lightweight stubs let unit tests exercise app logic before ML packages are installed.
for name in ("cv2", "torch", "transformers", "ultralytics"):
    sys.modules.setdefault(name, types.ModuleType(name))

from main import (
    AppSettings,
    apply_regional_prior,
    combine_classifier_predictions,
    CooldownTracker,
    EmailSettings,
    IdentificationResult,
    env_bool,
    hybrid_species_predictions,
    identification_plausible_species,
    format_species_name,
    open_camera,
    season_for_month,
    resolve_identification,
    regional_species,
    preferred_regional_species,
    preferred_regional_species_names,
    select_sharpest_crops,
    save_bird_image,
    send_email,
    validate_settings,
)


class FakeCV2:
    @staticmethod
    def imwrite(path, image):
        Path(path).write_bytes(b"jpeg")
        return True


class FakeSMTP:
    sent_message = None

    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def ehlo(self):
        pass

    def starttls(self, context):
        pass

    def login(self, username, password):
        self.login_args = (username, password)

    def send_message(self, message):
        FakeSMTP.sent_message = message


class BirdWatcherTests(unittest.TestCase):
    def test_season_is_derived_from_observation_month(self):
        self.assertEqual(season_for_month(1), "winter")
        self.assertEqual(season_for_month(4), "spring")
        self.assertEqual(season_for_month(7), "summer")
        self.assertEqual(season_for_month(10), "fall")

    def test_northern_nj_summer_prior_reranks_a_plausible_species(self):
        plausible = regional_species("northern_nj", 7)
        self.assertIn("northerncardinal", plausible)
        self.assertIn("brownheadedcowbird", plausible)
        self.assertGreaterEqual(len(plausible), 100)
        self.assertNotIn("oilbird", plausible)
        preferred = preferred_regional_species("northern_nj", 7)
        self.assertTrue(preferred <= plausible)
        predictions = [
            ("Azaras Spinetail", 0.55),
            ("Northern Cardinal", 0.30),
            ("House Finch", 0.15),
        ]
        adjusted = apply_regional_prior(predictions, preferred, weight=3.0)
        self.assertEqual(adjusted[0][0], "Northern Cardinal")
        self.assertAlmostEqual(sum(score for _, score in adjusted), 1.0)

    def test_broad_nj_species_receives_a_moderate_boost(self):
        plausible = regional_species("northern_nj", 7)
        adjusted = apply_regional_prior(
            [("Oilbird", 0.45), ("Osprey", 0.35), ("Azaras Spinetail", 0.20)],
            preferred_species=set(),
            weight=3.0,
            plausible_species=plausible,
            plausible_weight=1.5,
        )
        self.assertEqual(adjusted[0][0], "Osprey")

    def test_hybrid_classifier_blends_local_evidence_without_deleting_unusual_species(self):
        combined = combine_classifier_predictions(
            local_predictions=[("House Finch", 0.80), ("House Sparrow", 0.20)],
            global_predictions=[("African Firefinch", 0.70), ("House Finch", 0.30)],
            local_weight=0.65,
        )
        self.assertEqual(combined[0][0], "House Finch")
        self.assertAlmostEqual(sum(score for _, score in combined), 1.0)
        self.assertIn("African Firefinch", [name for name, _ in combined])
        self.assertAlmostEqual(dict(combined)["House Finch"], 0.625)
        self.assertAlmostEqual(dict(combined)["African Firefinch"], 0.245)

    def test_local_classifier_names_follow_the_active_season(self):
        summer = preferred_regional_species_names("northern_nj", 7)
        winter = preferred_regional_species_names("northern_nj", 1)
        self.assertIn("Northern Cardinal", summer)
        self.assertIn("Brown-Headed Cowbird", summer)
        self.assertNotIn("Brown-Headed Cowbird", winter)
        self.assertIn("Dark-Eyed Junco", winter)

    def test_local_species_missing_from_legacy_labels_still_pass_plausibility(self):
        broad = regional_species("northern_nj", 7)
        plausible = identification_plausible_species("northern_nj", 7)
        self.assertNotIn("carolinawren", broad)
        self.assertIn("carolinawren", plausible)

    def test_hybrid_path_uses_local_and_global_model_evidence(self):
        class FakeModels:
            def identify_species_candidates(self, bird_image, top_k=20):
                self.global_call = (bird_image, top_k)
                return [("African Firefinch", 0.70), ("House Finch", 0.30)]

            def identify_local_species_candidates(self, bird_image, species_names, top_k=20):
                self.local_call = (bird_image, species_names, top_k)
                return [("House Finch", 0.80), ("House Sparrow", 0.20)]

        models = FakeModels()
        predictions = hybrid_species_predictions(
            models=models,
            bird_image="crop",
            preferred_names={"House Finch", "House Sparrow"},
            preferred_keys={"housefinch", "housesparrow"},
            plausible_keys={"housefinch", "housesparrow"},
            regional_weight=3.0,
            local_weight=0.65,
        )
        self.assertEqual(predictions[0][0], "House Finch")
        self.assertEqual(models.global_call, ("crop", 20))
        self.assertEqual(models.local_call, ("crop", {"House Finch", "House Sparrow"}, 2))
        self.assertIn("African Firefinch", [name for name, _ in predictions])

    def test_consensus_accepts_agreeing_high_quality_predictions(self):
        result = resolve_identification(
            [
                [("Northern Cardinal", 0.92), ("Pyrrhuloxia", 0.04), ("House Finch", 0.02)],
                [("Northern Cardinal", 0.86), ("House Finch", 0.08), ("Pyrrhuloxia", 0.03)],
                [("House Finch", 0.51), ("Northern Cardinal", 0.43), ("Purple Finch", 0.03)],
            ],
            min_votes=2,
            min_confidence=0.70,
            min_margin=0.20,
        )
        self.assertFalse(result.uncertain)
        self.assertEqual(result.display_name, "Northern Cardinal")
        self.assertEqual(result.candidate_name, "Northern Cardinal")
        self.assertEqual(result.votes, 2)

    def test_consensus_marks_weak_evidence_uncertain(self):
        result = resolve_identification(
            [
                [("Northern Cardinal", 0.55), ("House Finch", 0.40)],
                [("House Finch", 0.52), ("Northern Cardinal", 0.44)],
                [("Northern Cardinal", 0.58), ("House Finch", 0.37)],
            ],
            min_votes=2,
            min_confidence=0.70,
            min_margin=0.20,
        )
        self.assertTrue(result.uncertain)
        self.assertEqual(result.candidate_name, "Northern Cardinal")
        self.assertEqual(result.display_name, "Uncertain bird")
        self.assertEqual(len(result.top_candidates), 2)

    def test_out_of_region_winner_is_not_reported_as_certain(self):
        result = resolve_identification(
            [[("Azaras Spinetail", 0.97), ("Northern Cardinal", 0.02)]] * 3,
            min_votes=2,
            min_confidence=0.70,
            min_margin=0.20,
            plausible_species=regional_species("northern_nj", 7),
        )
        self.assertTrue(result.uncertain)
        self.assertEqual(result.display_name, "Uncertain bird")

    def test_selects_the_three_sharpest_bird_crops(self):
        detections = [("soft", 0.9), ("sharp", 0.8), ("medium", 0.7), ("best", 0.6)]
        scores = {"soft": 10.0, "sharp": 80.0, "medium": 50.0, "best": 100.0}
        with patch("main.image_sharpness", side_effect=lambda image: scores[image]):
            selected = select_sharpest_crops(detections, 3)
        self.assertEqual([image for image, _ in selected], ["best", "sharp", "medium"])

    def test_linux_video_path_uses_v4l2_backend(self):
        class Capture:
            def __init__(self):
                self.set_calls = []

            def isOpened(self):
                return True

            def set(self, prop, value):
                self.set_calls.append((prop, value))
                return True

        class CameraCV2:
            CAP_V4L2 = 200
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            CAP_PROP_FPS = 5
            calls = []


            @classmethod
            def VideoCapture(cls, *args):
                cls.calls.append(args)
                return Capture()

        camera_path = "/dev/v4l/by-id/usb-camera-video-index0"
        with patch("main.cv2", CameraCV2):
            capture = open_camera(camera_path, 1280, 960, 5)
        self.assertIsNotNone(capture)
        self.assertEqual(CameraCV2.calls, [(camera_path, CameraCV2.CAP_V4L2)])
        self.assertEqual(
            capture.set_calls,
            [
                (CameraCV2.CAP_PROP_FRAME_WIDTH, 1280),
                (CameraCV2.CAP_PROP_FRAME_HEIGHT, 960),
                (CameraCV2.CAP_PROP_FPS, 5),
            ],
        )

    def test_cooldown_blocks_all_repeat_alerts_until_interval_expires(self):
        tracker = CooldownTracker(timedelta(minutes=10))
        now = datetime(2026, 7, 19, 12, 0, 0)
        self.assertTrue(tracker.is_ready("American Robin", now))
        tracker.mark_sent("American Robin", now)
        self.assertFalse(tracker.is_ready("American Robin", now + timedelta(minutes=9, seconds=59)))
        self.assertTrue(tracker.is_ready("American Robin", now + timedelta(minutes=10)))
        self.assertFalse(tracker.is_ready("Northern Cardinal", now + timedelta(minutes=1)))

    def test_cooldown_survives_a_service_restart(self):
        with TemporaryDirectory() as temp:
            state_path = Path(temp) / ".last_alert"
            now = datetime(2026, 7, 19, 12, 0, 0)
            first = CooldownTracker(timedelta(minutes=10), state_path)
            first.mark_sent("American Robin", now)
            restarted = CooldownTracker(timedelta(minutes=10), state_path)
            self.assertFalse(restarted.is_ready("Northern Cardinal", now + timedelta(minutes=1)))
            self.assertTrue(restarted.is_ready("Northern Cardinal", now + timedelta(minutes=10)))

    def test_species_name_is_human_readable(self):
        self.assertEqual(format_species_name("BLACK-CAPPED CHICKADEE"), "Black-Capped Chickadee")

    def test_save_bird_image_creates_directory_and_file(self):
        with TemporaryDirectory() as temp:
            with patch("main.cv2", FakeCV2):
                path = save_bird_image(object(), Path(temp) / "captures", "American Robin", datetime(2026, 7, 19, 12, 0, 0))
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "20260719_120000_000000_american_robin.jpg")

    def test_image_names_do_not_collide_within_one_second(self):
        with TemporaryDirectory() as temp:
            with patch("main.cv2", FakeCV2):
                first = save_bird_image(object(), Path(temp), "Robin", datetime(2026, 7, 19, 12, 0, 0, 1))
                second = save_bird_image(object(), Path(temp), "Robin", datetime(2026, 7, 19, 12, 0, 0, 2))
            self.assertNotEqual(first, second)

    def test_email_has_required_subject_body_and_attachment(self):
        with TemporaryDirectory() as temp:
            image = Path(temp) / "bird.jpg"
            image.write_bytes(b"jpeg")
            settings = EmailSettings("smtp.example.com", 587, "user", "secret", "from@example.com", "to@example.com", False, True, False)
            identification = IdentificationResult(
                display_name="American Robin",
                candidate_name="American Robin",
                confidence=0.934,
                margin=0.75,
                votes=3,
                frame_count=3,
                uncertain=False,
                top_candidates=(("American Robin", 0.934),),
            )
            with patch("main.smtplib.SMTP", FakeSMTP):
                send_email(settings, identification, datetime(2026, 7, 19, 12, 0, 0), image)
            message = FakeSMTP.sent_message
            self.assertEqual(message["Subject"], "Bird spotted: American Robin")
            body = message.get_body(preferencelist=("plain",)).get_content()
            self.assertIn("Bird: American Robin", body)
            self.assertIn("Confidence: 93.4%", body)
            self.assertIn("Time: 2026-07-19 12:00:00", body)
            self.assertEqual(message.get_payload()[-1].get_filename(), "bird.jpg")

    def test_uncertain_email_includes_an_explicit_approximate_guess(self):
        with TemporaryDirectory() as temp:
            image = Path(temp) / "bird.jpg"
            image.write_bytes(b"jpeg")
            settings = EmailSettings("smtp.example.com", 587, "user", "secret", "from@example.com", "to@example.com", False, True, False)
            identification = IdentificationResult(
                display_name="Uncertain bird",
                candidate_name="Northern Cardinal",
                confidence=0.55,
                margin=0.25,
                votes=3,
                frame_count=3,
                uncertain=True,
                top_candidates=(("Northern Cardinal", 0.55), ("House Finch", 0.30), ("House Sparrow", 0.15)),
            )
            with patch("main.smtplib.SMTP", FakeSMTP):
                send_email(settings, identification, datetime(2026, 7, 19, 12, 0, 0), image)
            message = FakeSMTP.sent_message
            self.assertEqual(
                message["Subject"],
                "Bird spotted: Uncertain bird (possible Northern Cardinal)",
            )
            body = message.get_body(preferencelist=("plain",)).get_content()
            self.assertIn("Identification: Uncertain bird", body)
            self.assertIn("Approximate guess: Northern Cardinal", body)
            self.assertIn("Approximate-guess score: 55.0%", body)
            self.assertIn("Agreement: 3 of 3 frames", body)
            self.assertIn("Top candidates: Northern Cardinal 55.0%, House Finch 30.0%, House Sparrow 15.0%", body)
            self.assertIn("This approximate guess did not meet the certainty requirements", body)

    def test_invalid_boolean_is_rejected(self):
        with patch.dict("os.environ", {"TEST_BOOLEAN": "tru"}):
            with self.assertRaisesRegex(ValueError, "TEST_BOOLEAN"):
                env_bool("TEST_BOOLEAN", False)

    def test_plaintext_smtp_requires_explicit_opt_in(self):
        email = EmailSettings("smtp.example.com", 25, "", "", "from@example.com", "to@example.com", False, False, False)
        settings = AppSettings(
            camera=1,
            camera_width=1280,
            camera_height=960,
            camera_fps=15,
            image_dir=Path("images"),
            cooldown=timedelta(minutes=10),
            scan_interval=1.0,
            detection_confidence=0.35,
            burst_frames=7,
            sharpest_frames=3,
            consensus_min_votes=2,
            species_min_confidence=0.70,
            species_min_margin=0.20,
            region_profile="northern_nj",
            regional_prior_weight=3.0,
            email=email,
            detector_model="model.pt",
            detector_sha256="a" * 64,
            classifier_model="classifier",
            classifier_revision="b" * 40,
            local_classifier_model="imageomics/bioclip",
            local_classifier_revision="c" * 40,
            local_classifier_weight=0.65,
        )
        with self.assertRaisesRegex(ValueError, "plaintext SMTP"):
            validate_settings(settings)

    def test_invalid_numeric_configuration_is_rejected(self):
        email = EmailSettings("smtp.example.com", 587, "user", "secret", "from@example.com", "to@example.com", False, True, False)
        settings = AppSettings(
            camera=1,
            camera_width=1280,
            camera_height=960,
            camera_fps=15,
            image_dir=Path("images"),
            cooldown=timedelta(0),
            scan_interval=-1.0,
            detection_confidence=1.5,
            burst_frames=7,
            sharpest_frames=3,
            consensus_min_votes=2,
            species_min_confidence=0.70,
            species_min_margin=0.20,
            region_profile="northern_nj",
            regional_prior_weight=3.0,
            email=email,
            detector_model="model.pt",
            detector_sha256="a" * 64,
            classifier_model="classifier",
            classifier_revision="b" * 40,
            local_classifier_model="imageomics/bioclip",
            local_classifier_revision="c" * 40,
            local_classifier_weight=0.65,
        )
        with self.assertRaises(ValueError):
            validate_settings(settings)


if __name__ == "__main__":
    unittest.main()
