"""Current Bird Watcher application runtime."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

import legacy_main as core

from .classification import encode_accepted_crops, hybrid_predictions
from .config import RuntimeConfig, load_runtime_config
from .tracking import (
    ActiveEventTracker,
    choose_initial_detection,
    collect_tracked_crops,
    detect_birds,
)

LOG = logging.getLogger("bird_watcher")


def process_new_event(
    capture,
    models,
    initial,
    settings,
    active_events: ActiveEventTracker,
    event_time: datetime,
    *,
    clock_fn: Callable[[], float] = time.monotonic,
) -> bool:
    """Validate, classify, save, and alert once for one newly observed bird event."""
    tracked = collect_tracked_crops(capture, models, initial, settings)
    valid, reason = core.validate_bird_event(
        [(item.crop, item.score) for item in tracked],
        settings.min_valid_bird_frames,
        settings.min_event_detector_confidence,
        settings.max_bird_crop_aspect_ratio,
    )
    if not valid:
        LOG.info("Suppressed false bird event: %s", reason)
        return False

    valid_ids = {id(crop) for crop, _ in valid}
    valid_tracked = [item for item in tracked if id(item.crop) in valid_ids]
    encoded = encode_accepted_crops(
        valid_tracked,
        models,
        settings.min_bird_presence_score,
    )
    if len(encoded) < settings.min_bird_presence_frames:
        LOG.info(
            "Suppressed bird event: only %d crop(s) passed BioCLIP presence gate",
            len(encoded),
        )
        return False

    # The event is now confirmed as a real bird. Timestamp it after the burst and
    # presence gate, not at the first scan, so a long inference cycle cannot make
    # the event look expired immediately on the next camera frame.
    previous_box = encoded[-2].detection.box if len(encoded) > 1 else None
    active_events.mark_event(
        encoded[-1].detection.box,
        clock_fn(),
        previous_box,
    )

    encoded.sort(
        key=lambda item: core.image_sharpness(item.detection.crop),
        reverse=True,
    )
    selected = encoded[: settings.sharpest_frames]
    preferred_names = core.preferred_regional_species_names(
        settings.region_profile, event_time.month
    )
    preferred = core.preferred_regional_species(
        settings.region_profile, event_time.month
    )
    broad = core.regional_species(settings.region_profile, event_time.month)
    plausible = core.identification_plausible_species(
        settings.region_profile, event_time.month
    )

    frame_predictions = [
        hybrid_predictions(
            models,
            item.detection.crop,
            item.embedding,
            preferred_names,
            preferred,
            broad,
            settings,
        )
        for item in selected
    ]
    identification = core.resolve_identification(
        frame_predictions,
        settings.consensus_min_votes,
        settings.species_min_confidence,
        settings.species_min_margin,
        plausible,
    )

    image_path = core.save_bird_image(
        selected[0].detection.crop,
        settings.image_dir,
        identification.candidate_name,
        event_time,
    )
    LOG.info(
        "Bird identification: %s; votes %d/%d, confidence %.1f%%, margin %.1f%%",
        identification.display_name,
        identification.votes,
        identification.frame_count,
        identification.confidence * 100,
        identification.margin * 100,
    )
    try:
        core.send_email(settings.email, identification, event_time, image_path)
    except Exception:
        LOG.exception(
            "Email failed; image remains saved and the active event stays suppressed"
        )
    else:
        LOG.info("Email alert sent for %s", identification.display_name)
    return True


def run(runtime_config: RuntimeConfig) -> None:
    settings = runtime_config.settings
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    models = core.BirdModels(
        settings.detector_model,
        settings.detector_sha256,
        settings.classifier_model,
        settings.classifier_revision,
        settings.local_classifier_model,
        settings.local_classifier_revision,
    )
    active_events = ActiveEventTracker(
        clear_seconds=runtime_config.event_clear_seconds,
        max_age_seconds=runtime_config.active_event_max_age.total_seconds(),
    )
    capture = None
    LOG.info(
        "Watching camera %r; active events clear after %.1fs of absence",
        settings.camera,
        runtime_config.event_clear_seconds,
    )

    try:
        while True:
            if capture is None:
                capture = core.open_camera(
                    settings.camera,
                    settings.camera_width,
                    settings.camera_height,
                    settings.camera_fps,
                )
                if capture is None:
                    LOG.error(
                        "Could not open camera %r; retrying in 5 seconds",
                        settings.camera,
                    )
                    time.sleep(5)
                    continue

            ok, frame = capture.read()
            if not ok or frame is None:
                LOG.error("Camera read failed; reopening camera")
                capture.release()
                capture = None
                time.sleep(2)
                continue

            try:
                scan_clock = time.monotonic()
                detections = detect_birds(models, frame, settings)
                if not detections:
                    active_events.observe_no_detection(scan_clock)
                    time.sleep(max(0.0, settings.scan_interval))
                    continue

                # Existing birds are touched and removed from consideration here,
                # before the expensive burst and classification work. A different
                # spatially distinct bird can still start a new event immediately.
                new_detections = active_events.partition_new_detections(
                    detections, scan_clock
                )
                initial = choose_initial_detection(new_detections)
                if initial is None:
                    LOG.debug(
                        "Only active bird event(s) detected; skipping extended classification"
                    )
                    time.sleep(max(0.0, settings.scan_interval))
                    continue

                process_new_event(
                    capture,
                    models,
                    initial,
                    settings,
                    active_events,
                    datetime.now().astimezone(),
                )
            except Exception:
                LOG.exception("Detection failed; continuing")

            time.sleep(max(0.0, settings.scan_interval))
    except KeyboardInterrupt:
        LOG.info("Stopping bird watcher")
    finally:
        if capture is not None:
            capture.release()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run(load_runtime_config())
    except Exception:
        LOG.exception("Bird watcher could not start")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
