"""Configuration loading for the current Bird Watcher runtime."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

import legacy_main as core


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated application settings plus event-deduplication controls."""

    settings: core.AppSettings
    event_clear_seconds: float
    active_event_max_age: timedelta


def load_runtime_config() -> RuntimeConfig:
    """Load .env without pre-populating values that would block dotenv overrides."""
    load_dotenv()

    camera_value = os.getenv("CAMERA", "0").strip()
    camera: int | str = int(camera_value) if camera_value.isdigit() else camera_value
    email = core.EmailSettings(
        host=os.getenv("SMTP_HOST", "").strip(),
        port=int(os.getenv("SMTP_PORT", "587")),
        username=os.getenv("SMTP_USERNAME", "").strip(),
        password=os.getenv("SMTP_PASSWORD", ""),
        sender=os.getenv("EMAIL_FROM", "").strip(),
        recipient=os.getenv("EMAIL_TO", "").strip(),
        use_ssl=core.env_bool("SMTP_USE_SSL", False),
        use_starttls=core.env_bool("SMTP_USE_STARTTLS", True),
        allow_insecure=core.env_bool("SMTP_ALLOW_INSECURE", False),
    )
    missing = [
        name
        for name, value in {
            "SMTP_HOST": email.host,
            "EMAIL_FROM": email.sender,
            "EMAIL_TO": email.recipient,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required .env settings: {', '.join(missing)}")

    # COOLDOWN_MINUTES remains a backwards-compatible alias for the maximum age
    # of one continuously active bird event. It is no longer keyed by species.
    max_event_minutes = float(
        os.getenv("ACTIVE_EVENT_MAX_MINUTES", os.getenv("COOLDOWN_MINUTES", "10"))
    )
    event_clear_seconds = float(os.getenv("EVENT_CLEAR_SECONDS", "3"))
    if not math.isfinite(max_event_minutes) or max_event_minutes <= 0:
        raise ValueError("ACTIVE_EVENT_MAX_MINUTES must be finite and greater than zero")
    if not math.isfinite(event_clear_seconds) or event_clear_seconds <= 0:
        raise ValueError("EVENT_CLEAR_SECONDS must be finite and greater than zero")

    settings = core.AppSettings(
        camera=camera,
        camera_width=int(os.getenv("CAMERA_WIDTH", "1280")),
        camera_height=int(os.getenv("CAMERA_HEIGHT", "960")),
        camera_fps=float(os.getenv("CAMERA_FPS", "5")),
        image_dir=Path(os.getenv("BIRD_IMAGE_DIR", "bird_images")).expanduser(),
        cooldown=timedelta(minutes=max_event_minutes),
        scan_interval=float(os.getenv("SCAN_INTERVAL_SECONDS", "1")),
        detection_confidence=float(os.getenv("DETECTION_CONFIDENCE", "0.35")),
        crop_padding=float(os.getenv("DETECTION_CROP_PADDING", "0.20")),
        burst_frames=int(os.getenv("BURST_FRAMES", "9")),
        burst_frame_interval=float(os.getenv("BURST_FRAME_INTERVAL_SECONDS", "1.0")),
        sharpest_frames=int(os.getenv("SHARPEST_FRAMES", "7")),
        min_valid_bird_frames=int(os.getenv("MIN_VALID_BIRD_FRAMES", "4")),
        min_event_detector_confidence=float(
            os.getenv("MIN_EVENT_DETECTOR_CONFIDENCE", "0.07")
        ),
        detection_floor_confidence=float(os.getenv("DETECTION_FLOOR_CONFIDENCE", "0.05")),
        max_bird_crop_aspect_ratio=float(os.getenv("MAX_BIRD_CROP_ASPECT_RATIO", "2.5")),
        min_bird_presence_score=float(os.getenv("MIN_BIRD_PRESENCE_SCORE", "0.50")),
        min_bird_presence_frames=int(os.getenv("MIN_BIRD_PRESENCE_FRAMES", "2")),
        consensus_min_votes=int(os.getenv("CONSENSUS_MIN_VOTES", "4")),
        species_min_confidence=float(os.getenv("SPECIES_MIN_CONFIDENCE", "0.60")),
        species_min_margin=float(os.getenv("SPECIES_MIN_MARGIN", "0.20")),
        region_profile=os.getenv("REGION_PROFILE", "northern_nj").strip(),
        regional_prior_weight=float(os.getenv("REGIONAL_PRIOR_WEIGHT", "3.0")),
        email=email,
        detector_model=os.getenv("DETECTOR_MODEL", "yolo11n.pt"),
        detector_sha256=os.getenv(
            "DETECTOR_MODEL_SHA256",
            "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        ),
        classifier_model=os.getenv("CLASSIFIER_MODEL", "chriamue/bird-species-classifier"),
        classifier_revision=os.getenv(
            "CLASSIFIER_REVISION",
            "558944ca4448f5b311af8393c8b894eff20a06da",
        ),
        local_classifier_model=os.getenv("LOCAL_CLASSIFIER_MODEL", "imageomics/bioclip"),
        local_classifier_revision=os.getenv(
            "LOCAL_CLASSIFIER_REVISION",
            "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c",
        ),
        local_classifier_weight=float(os.getenv("LOCAL_CLASSIFIER_WEIGHT", "0.65")),
    )
    core.validate_settings(settings)
    return RuntimeConfig(
        settings=settings,
        event_clear_seconds=event_clear_seconds,
        active_event_max_age=timedelta(minutes=max_event_minutes),
    )
