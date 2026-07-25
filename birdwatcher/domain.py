"""Shared immutable data types for the Bird Watcher runtime."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path


@dataclass(frozen=True)
class EmailSettings:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipient: str
    use_ssl: bool
    use_starttls: bool
    allow_insecure: bool


@dataclass(frozen=True)
class AppSettings:
    camera: int | str
    camera_width: int
    camera_height: int
    camera_fps: float
    image_dir: Path
    cooldown: timedelta
    scan_interval: float
    detection_confidence: float
    tile_sweep_interval: float
    crop_padding: float
    burst_frames: int
    burst_frame_interval: float
    sharpest_frames: int
    min_valid_bird_frames: int
    min_event_detector_confidence: float
    detection_floor_confidence: float
    max_bird_crop_aspect_ratio: float
    min_bird_presence_score: float
    min_bird_presence_frames: int
    consensus_min_votes: int
    species_min_confidence: float
    species_min_margin: float
    region_profile: str
    regional_prior_weight: float
    email: EmailSettings
    detector_model: str
    detector_sha256: str
    classifier_model: str
    classifier_revision: str
    local_classifier_model: str
    local_classifier_revision: str
    local_classifier_weight: float


@dataclass(frozen=True)
class IdentificationResult:
    display_name: str
    candidate_name: str
    confidence: float
    margin: float
    votes: int
    frame_count: int
    uncertain: bool
    top_candidates: tuple[tuple[str, float], ...]
