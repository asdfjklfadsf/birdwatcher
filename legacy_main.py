"""Compatibility facade for pre-refactor Bird Watcher imports.

Production code lives entirely under :mod:`birdwatcher` and does not depend on
this module. Older launchers may still execute ``python legacy_main.py`` while
migrating to the canonical ``python main.py`` entrypoint.
"""
from __future__ import annotations

from birdwatcher.app import main, process_new_event, run
from birdwatcher.domain import AppSettings, EmailSettings, IdentificationResult
from birdwatcher.emailer import send_email
from birdwatcher.media import image_sharpness, open_camera, safe_filename, save_bird_image
from birdwatcher.models import BirdModels, file_sha256, format_species_name, prepare_detector_model
from birdwatcher.region import (
    apply_regional_prior,
    combine_classifier_predictions,
    identification_plausible_species,
    preferred_regional_species,
    preferred_regional_species_names,
    regional_species,
    resolve_identification,
    season_for_month,
    species_key,
)
from birdwatcher.settings import env_bool, validate_settings
from birdwatcher.validation import validate_bird_event, validate_visual_bird_presence


def select_sharpest_crops(detections: list[tuple[object, float]], count: int):
    if count <= 0:
        raise ValueError("Sharpest crop count must be positive")
    return sorted(detections, key=lambda item: image_sharpness(item[0]), reverse=True)[:count]


if __name__ == "__main__":
    main()
