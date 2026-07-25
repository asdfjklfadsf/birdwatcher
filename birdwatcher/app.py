"""Current Bird Watcher application runtime."""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

from .alerts import EmailRetryQueue, send_or_queue
from .classification import encode_accepted_crops, hybrid_predictions
from .config import RuntimeConfig, load_runtime_config
from .emailer import send_email
from .media import image_sharpness, open_camera, save_bird_image
from .models import BirdModels
from .region import (
    identification_plausible_species,
    preferred_regional_species,
    preferred_regional_species_names,
    regional_species,
    resolve_identification,
)
from .tracking import (
    ActiveEventTracker,
    TileSweepThrottle,
    choose_initial_detection,
    collect_tracked_crops,
    detect_birds,
)
from .validation import validate_bird_event

LOG = logging.getLogger("bird_watcher")


def process_new_event(
    capture,
    models,
    initial,
    settings,
    active_events: ActiveEventTracker,
    event_time: datetime,
    *,
    retry_queue: EmailRetryQueue | None = None,
    clock_fn: Callable[[], float] = time.monotonic,
    tile_sweep: TileSweepThrottle | None = None,
) -> bool:
    """Validate, classify, save, and alert once for one newly observed bird event."""
    tracked = collect_tracked_crops(capture, models, initial, settings, tile_sweep=tile_sweep)
    valid, reason = validate_bird_event(
        [(item.crop, item.score) for item in tracked],
        settings.min_valid_bird_frames,
        settings.min_event_detector_confidence,
        settings.max_bird_crop_aspect_ratio,
    )
    if not valid:
        LOG.info("Suppressed false bird event: %s", reason)
        return False

    # validate_bird_event filters the (crop, score) pairs without copying the
    # crops, so the accepted crop objects are the very ones held by `tracked`.
    # Identity is what maps them back; equality would be wrong for image arrays.
    valid_crops = {id(crop) for crop, _ in valid}
    valid_tracked = [item for item in tracked if id(item.crop) in valid_crops]
    encoded = encode_accepted_crops(valid_tracked, models, settings.min_bird_presence_score)
    if len(encoded) < settings.min_bird_presence_frames:
        LOG.info("Suppressed bird event: only %d crop(s) passed BioCLIP presence gate", len(encoded))
        return False

    confirmed_box = encoded[-1].detection.box
    previous_box = encoded[-2].detection.box if len(encoded) > 1 else None
    active_events.mark_event(confirmed_box, clock_fn(), previous_box)

    try:
        encoded.sort(key=lambda item: image_sharpness(item.detection.crop), reverse=True)
        selected = encoded[: settings.sharpest_frames]
        preferred_names = preferred_regional_species_names(settings.region_profile, event_time.month)
        preferred = preferred_regional_species(settings.region_profile, event_time.month)
        broad = regional_species(settings.region_profile, event_time.month)
        plausible = identification_plausible_species(settings.region_profile, event_time.month)

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
        identification = resolve_identification(
            frame_predictions,
            settings.consensus_min_votes,
            settings.species_min_confidence,
            settings.species_min_margin,
            plausible,
        )

        image_path = save_bird_image(
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
        if retry_queue is None:
            try:
                send_email(settings.email, identification, event_time, image_path)
            except Exception:
                LOG.exception("Email failed; image remains saved")
            else:
                LOG.info("Email alert sent for %s", identification.display_name)
        elif send_or_queue(retry_queue, settings.email, identification, event_time, image_path):
            LOG.info("Email alert sent for %s", identification.display_name)
        return True
    finally:
        # Classification and SMTP can take longer than EVENT_CLEAR_SECONDS on CPU.
        # Refresh from completion time so a continuously present bird is not treated
        # as a new event immediately after the expensive processing path returns.
        active_events.mark_event(confirmed_box, clock_fn(), previous_box)


def run(runtime_config: RuntimeConfig) -> None:
    settings = runtime_config.settings
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    models = BirdModels(
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
    retry_queue = EmailRetryQueue(settings.image_dir / ".email_retry_queue")
    capture = None
    tile_sweep = TileSweepThrottle(settings.tile_sweep_interval)
    LOG.info(
        "Watching camera %r; active events clear after %.1fs of absence",
        settings.camera,
        runtime_config.event_clear_seconds,
    )

    try:
        while True:
            retry_queue.retry_due(settings.email)
            if capture is None:
                capture = open_camera(
                    settings.camera,
                    settings.camera_width,
                    settings.camera_height,
                    settings.camera_fps,
                )
                if capture is None:
                    LOG.error("Could not open camera %r; retrying in 5 seconds", settings.camera)
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
                # The nine-tile sweep only helps for small/distant birds and
                # costs nine extra inferences, so run it on a slower cadence
                # instead of on every idle frame. Tracked bursts share this
                # budget rather than getting an unthrottled one of their own.
                detections = detect_birds(
                    models, frame, settings, allow_tile_sweep=tile_sweep.allow(scan_clock)
                )
                if not detections:
                    active_events.observe_no_detection(scan_clock)
                    time.sleep(max(0.0, settings.scan_interval))
                    continue

                new_detections = active_events.partition_new_detections(detections, scan_clock)
                initial = choose_initial_detection(new_detections)
                if initial is None:
                    LOG.debug("Only active bird event(s) detected; skipping extended classification")
                    time.sleep(max(0.0, settings.scan_interval))
                    continue

                process_new_event(
                    capture,
                    models,
                    initial,
                    settings,
                    active_events,
                    datetime.now().astimezone(),
                    retry_queue=retry_queue,
                    tile_sweep=tile_sweep,
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        run(load_runtime_config())
    except Exception:
        LOG.exception("Bird watcher could not start")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
