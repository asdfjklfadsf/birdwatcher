"""Validation helpers for application settings."""
from __future__ import annotations

import math
import os
import re
from datetime import timedelta

from .constants import BROAD_REGIONAL_PRIOR_WEIGHT
from .domain import AppSettings
from .region import regional_species


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, not {value!r}")


def validate_settings(settings: AppSettings) -> None:
    if settings.camera_width <= 0 or settings.camera_height <= 0:
        raise ValueError("CAMERA_WIDTH and CAMERA_HEIGHT must be positive")
    if not math.isfinite(settings.camera_fps) or settings.camera_fps <= 0:
        raise ValueError("CAMERA_FPS must be finite and greater than zero")
    if not 1 <= settings.email.port <= 65535:
        raise ValueError("SMTP_PORT must be between 1 and 65535")
    if settings.cooldown <= timedelta(0):
        raise ValueError("ACTIVE_EVENT_MAX_MINUTES must be greater than zero")
    if not math.isfinite(settings.scan_interval) or settings.scan_interval < 0:
        raise ValueError("SCAN_INTERVAL_SECONDS must be finite and nonnegative")
    if not math.isfinite(settings.detection_confidence) or not 0 < settings.detection_confidence <= 1:
        raise ValueError("DETECTION_CONFIDENCE must be greater than 0 and at most 1")
    if not math.isfinite(settings.crop_padding) or not 0 <= settings.crop_padding <= 1:
        raise ValueError("DETECTION_CROP_PADDING must be between 0 and 1")
    if settings.burst_frames < 1:
        raise ValueError("BURST_FRAMES must be at least 1")
    if not math.isfinite(settings.burst_frame_interval) or settings.burst_frame_interval < 0:
        raise ValueError("BURST_FRAME_INTERVAL_SECONDS must be finite and nonnegative")
    if not 1 <= settings.sharpest_frames <= settings.burst_frames:
        raise ValueError("SHARPEST_FRAMES must be between 1 and BURST_FRAMES")
    if not 1 <= settings.min_valid_bird_frames <= settings.burst_frames:
        raise ValueError("MIN_VALID_BIRD_FRAMES must be between 1 and BURST_FRAMES")
    if not 0 < settings.min_event_detector_confidence <= 1:
        raise ValueError("MIN_EVENT_DETECTOR_CONFIDENCE must be greater than 0 and at most 1")
    if not 0 < settings.detection_floor_confidence <= 1:
        raise ValueError("DETECTION_FLOOR_CONFIDENCE must be greater than 0 and at most 1")
    if settings.detection_floor_confidence > settings.min_event_detector_confidence:
        raise ValueError("DETECTION_FLOOR_CONFIDENCE must not exceed MIN_EVENT_DETECTOR_CONFIDENCE")
    if not math.isfinite(settings.max_bird_crop_aspect_ratio) or settings.max_bird_crop_aspect_ratio < 1:
        raise ValueError("MAX_BIRD_CROP_ASPECT_RATIO must be finite and at least 1")
    if not 0 <= settings.min_bird_presence_score <= 1:
        raise ValueError("MIN_BIRD_PRESENCE_SCORE must be between 0 and 1")
    if not 1 <= settings.min_bird_presence_frames <= settings.min_valid_bird_frames:
        raise ValueError("MIN_BIRD_PRESENCE_FRAMES must be between 1 and MIN_VALID_BIRD_FRAMES")
    if not 1 <= settings.consensus_min_votes <= settings.sharpest_frames:
        raise ValueError("CONSENSUS_MIN_VOTES must be between 1 and SHARPEST_FRAMES")
    if not 0 <= settings.species_min_confidence <= 1:
        raise ValueError("SPECIES_MIN_CONFIDENCE must be between 0 and 1")
    if not 0 <= settings.species_min_margin <= 1:
        raise ValueError("SPECIES_MIN_MARGIN must be between 0 and 1")
    if (
        not math.isfinite(settings.regional_prior_weight)
        or settings.regional_prior_weight < BROAD_REGIONAL_PRIOR_WEIGHT
    ):
        raise ValueError(
            "REGIONAL_PRIOR_WEIGHT must be finite and at least "
            f"{BROAD_REGIONAL_PRIOR_WEIGHT:g}"
        )
    regional_species(settings.region_profile, 7)
    if settings.email.use_ssl and settings.email.use_starttls:
        raise ValueError("Enable only one of SMTP_USE_SSL and SMTP_USE_STARTTLS")
    if not settings.email.use_ssl and not settings.email.use_starttls and not settings.email.allow_insecure:
        raise ValueError("Refusing plaintext SMTP unless SMTP_ALLOW_INSECURE=true")
    if bool(settings.email.username) != bool(settings.email.password):
        raise ValueError("SMTP_USERNAME and SMTP_PASSWORD must either both be set or both be empty")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", settings.detector_sha256):
        raise ValueError("DETECTOR_MODEL_SHA256 must contain exactly 64 hexadecimal characters")
    if not settings.classifier_revision:
        raise ValueError("CLASSIFIER_REVISION is required")
    if not settings.local_classifier_model or not settings.local_classifier_revision:
        raise ValueError("LOCAL_CLASSIFIER_MODEL and LOCAL_CLASSIFIER_REVISION are required")
    if not math.isfinite(settings.local_classifier_weight) or not 0 <= settings.local_classifier_weight <= 1:
        raise ValueError("LOCAL_CLASSIFIER_WEIGHT must be between 0 and 1")
