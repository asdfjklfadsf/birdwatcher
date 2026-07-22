"""Improved Bird Watcher runtime with single-bird tracking and broader BioCLIP reranking.

This module reuses the existing project's settings, models, persistence helpers, and email
formatting while fixing event association and candidate-scoring behavior. Run it with:

    python birdwatcher_improved.py
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Keep the documented fallback aligned even when .env omits it.
os.environ.setdefault("MIN_EVENT_DETECTOR_CONFIDENCE", "0.07")

import cv2
import main as legacy

LOG = logging.getLogger("bird_watcher")


@dataclass(frozen=True)
class TrackedDetection:
    crop: Any
    score: float
    box: tuple[int, int, int, int]


@dataclass(frozen=True)
class EncodedCrop:
    detection: TrackedDetection
    embedding: Any
    presence_score: float


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    intersection = iw * ih
    if not intersection:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def center_distance_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    distance = math.hypot(acx - bcx, acy - bcy)
    scale = max(1.0, math.hypot(max(ax2 - ax1, bx2 - bx1), max(ay2 - ay1, by2 - by1)))
    return distance / scale


def pad_crop(frame, box: tuple[int, int, int, int], crop_padding: float):
    x1, y1, x2, y2 = box
    height, width = frame.shape[:2]
    box_dim = max(x2 - x1, y2 - y1)
    relative_padding = min(1.0, crop_padding + 0.6 * (1.0 - min(1.0, box_dim / 220.0)))
    padding = int(relative_padding * box_dim)
    px1, py1 = max(0, x1 - padding), max(0, y1 - padding)
    px2, py2 = min(width, x2 + padding), min(height, y2 + padding)
    crop = frame[py1:py2, px1:px2].copy()
    if crop.size == 0:
        raise ValueError("Detector returned an empty bird crop")
    return crop


def detect_birds(models, frame, settings) -> list[TrackedDetection]:
    """Return all plausible bird detections in full-frame coordinates."""
    height, width = frame.shape[:2]
    raw: list[tuple[int, int, int, int, float]] = []

    def accept(x1: int, y1: int, x2: int, y2: int, score: float) -> None:
        box_width, box_height = max(0, x2 - x1), max(0, y2 - y1)
        if not box_width or not box_height:
            return
        ratio = max(box_width / box_height, box_height / box_width)
        if ratio <= settings.max_bird_crop_aspect_ratio:
            raw.append((x1, y1, x2, y2, score))

    for x1, y1, x2, y2, score in models._collect_bird_boxes(
        frame, 640, settings.detection_confidence
    ):
        accept(x1, y1, x2, y2, score)

    if not raw:
        for x1, y1, x2, y2, score in models._collect_bird_boxes(
            frame, 1280, settings.detection_floor_confidence
        ):
            accept(x1, y1, x2, y2, score)
        for ty in range(3):
            for tx in range(3):
                y0, y1t = int(ty * height / 3), int((ty + 1) * height / 3)
                x0, x1t = int(tx * width / 3), int((tx + 1) * width / 3)
                tile = frame[y0:y1t, x0:x1t]
                if tile.size == 0:
                    continue
                for lx1, ly1, lx2, ly2, score in models._collect_bird_boxes(
                    tile, 640, settings.detection_floor_confidence
                ):
                    accept(x0 + lx1, y0 + ly1, x0 + lx2, y0 + ly2, score)

    return [
        TrackedDetection(pad_crop(frame, (x1, y1, x2, y2), settings.crop_padding), score, (x1, y1, x2, y2))
        for x1, y1, x2, y2, score in raw
    ]


def choose_initial_detection(detections: list[TrackedDetection]) -> TrackedDetection | None:
    if not detections:
        return None
    return max(
        detections,
        key=lambda item: (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]) * item.score,
    )


def match_tracked_detection(
    previous_box: tuple[int, int, int, int],
    detections: list[TrackedDetection],
    min_iou: float = 0.05,
    max_center_distance_ratio: float = 1.25,
) -> TrackedDetection | None:
    """Match the same bird without silently switching to a different subject."""
    candidates = []
    for detection in detections:
        iou = box_iou(previous_box, detection.box)
        distance = center_distance_ratio(previous_box, detection.box)
        if iou >= min_iou or distance <= max_center_distance_ratio:
            candidates.append((iou - 0.25 * distance, detection.score, detection))
    return max(candidates, default=(None, None, None), key=lambda item: (item[0], item[1]))[2]


def collect_tracked_crops(capture, models, initial_detection: TrackedDetection, settings):
    detections = [initial_detection]
    previous_box = initial_detection.box
    next_sample = time.monotonic() + settings.burst_frame_interval
    for _ in range(settings.burst_frames - 1):
        delay = next_sample - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        next_sample += settings.burst_frame_interval
        ok, frame = capture.read()
        if not ok or frame is None:
            LOG.warning("Camera read failed during tracked bird event; skipping frame")
            continue
        matched = match_tracked_detection(previous_box, detect_birds(models, frame, settings))
        if matched is None:
            LOG.debug("Tracked bird not matched in observation frame")
            continue
        detections.append(matched)
        previous_box = matched.box
    return detections


def encode_bioclip_image(models, bird_image):
    from PIL import Image

    rgb_image = cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB)
    tensor = models.local_preprocess(Image.fromarray(rgb_image)).unsqueeze(0).to(models.device)
    with models.torch.inference_mode():
        features = models.local_classifier.encode_image(tensor)
        return features / features.norm(dim=-1, keepdim=True)


def text_features(models, names: tuple[str, ...]):
    cache = getattr(models, "_improved_text_cache", None)
    if cache is None:
        cache = {}
        models._improved_text_cache = cache
    if names in cache:
        return cache[names]
    templates = (
        "a photo of a {}.",
        "a cropped photo of a {}.",
        "a close-up photo of a {}.",
        "a photo of the {}.",
        "a blurry photo of a {}.",
    )
    prompts = [template.format(name) for name in names for template in templates]
    with models.torch.inference_mode():
        features = models.local_classifier.encode_text(models.local_tokenizer(prompts).to(models.device))
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.reshape(len(names), len(templates), -1).mean(dim=1)
        features = features / features.norm(dim=-1, keepdim=True)
        cache[names] = features.T.contiguous()
    return cache[names]


def bird_presence_from_embedding(models, embedding) -> float:
    features = getattr(models, "_improved_presence_text_features", None)
    if features is None:
        groups = (
            (
                "a photograph of a real bird at a bird feeder",
                "a real bird with feathers, head, beak, body and legs",
                "a perched wild bird",
                "a small bird eating seeds",
            ),
            (
                "an empty bird feeder with no bird",
                "a clear plastic feeder dish and seeds",
                "bird feeder hardware with no animal",
                "an empty feeding tray, foliage and reflections",
            ),
        )
        prompts = [prompt for group in groups for prompt in group]
        with models.torch.inference_mode():
            features = models.local_classifier.encode_text(models.local_tokenizer(prompts).to(models.device))
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.reshape(2, len(groups[0]), -1).mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.T.contiguous()
        models._improved_presence_text_features = features
    with models.torch.inference_mode():
        probabilities = models.torch.softmax(100.0 * embedding @ features, dim=-1)[0]
    return float(probabilities[0])


def local_predictions_from_embedding(models, embedding, species_names: set[str]):
    names = tuple(sorted(species_names))
    if not names:
        raise ValueError("BioCLIP requires at least one candidate species")
    features = text_features(models, names)
    with models.torch.inference_mode():
        logits = models.local_classifier.logit_scale.exp() * embedding @ features
        probabilities = models.torch.softmax(logits, dim=-1)[0]
    return sorted(
        ((name, float(probabilities[index])) for index, name in enumerate(names)),
        key=lambda item: -item[1],
    )


def hybrid_predictions(models, bird_image, embedding, preferred_names, preferred_keys, broad_keys, settings):
    raw_global = models.identify_species_candidates(bird_image, top_k=20)
    global_predictions = legacy.apply_regional_prior(
        raw_global,
        preferred_keys,
        settings.regional_prior_weight,
        plausible_species=broad_keys,
        plausible_weight=1.5,
    )
    candidate_names = set(preferred_names) | {name for name, _ in raw_global}
    local_predictions = local_predictions_from_embedding(models, embedding, candidate_names)
    return legacy.combine_classifier_predictions(
        local_predictions, global_predictions, settings.local_classifier_weight
    )


class SpeciesCooldownTracker:
    """Persist per-species cooldowns so a different species can alert immediately."""

    def __init__(self, cooldown: timedelta, state_path: Path):
        self.cooldown = cooldown
        self.state_path = state_path
        self._last_sent: dict[str, datetime] = {}
        if state_path.exists():
            try:
                payload = json.loads(state_path.read_text())
                self._last_sent = {key: datetime.fromisoformat(value) for key, value in payload.items()}
            except (OSError, ValueError, TypeError):
                LOG.warning("Ignoring invalid species cooldown state file: %s", state_path)

    def is_ready(self, species: str, now: datetime) -> bool:
        last = self._last_sent.get(legacy.species_key(species))
        return last is None or now - last >= self.cooldown

    def mark_sent(self, species: str, now: datetime) -> None:
        self._last_sent[legacy.species_key(species)] = now
        temporary = self.state_path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps({key: value.isoformat() for key, value in self._last_sent.items()}, sort_keys=True)
            )
            temporary.replace(self.state_path)
        except OSError:
            LOG.exception("Could not persist species cooldown state to %s", self.state_path)
            temporary.unlink(missing_ok=True)


def run(settings) -> None:
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    models = legacy.BirdModels(
        settings.detector_model,
        settings.detector_sha256,
        settings.classifier_model,
        settings.classifier_revision,
        settings.local_classifier_model,
        settings.local_classifier_revision,
    )
    cooldown = SpeciesCooldownTracker(settings.cooldown, settings.image_dir / ".species_alerts.json")
    capture = None
    LOG.info("Watching camera %r with tracked-event runtime", settings.camera)

    try:
        while True:
            if capture is None:
                capture = legacy.open_camera(
                    settings.camera, settings.camera_width, settings.camera_height, settings.camera_fps
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
                initial = choose_initial_detection(detect_birds(models, frame, settings))
                if initial is None:
                    time.sleep(max(0.0, settings.scan_interval))
                    continue

                now = datetime.now().astimezone()
                tracked = collect_tracked_crops(capture, models, initial, settings)
                valid, reason = legacy.validate_bird_event(
                    [(item.crop, item.score) for item in tracked],
                    settings.min_valid_bird_frames,
                    settings.min_event_detector_confidence,
                    settings.max_bird_crop_aspect_ratio,
                )
                if not valid:
                    LOG.info("Suppressed false bird event: %s", reason)
                    continue

                valid_ids = {id(crop) for crop, _ in valid}
                encoded: list[EncodedCrop] = []
                for detection in tracked:
                    if id(detection.crop) not in valid_ids:
                        continue
                    embedding = encode_bioclip_image(models, detection.crop)
                    presence_score = bird_presence_from_embedding(models, embedding)
                    if presence_score >= settings.min_bird_presence_score:
                        encoded.append(EncodedCrop(detection, embedding, presence_score))
                if len(encoded) < settings.min_bird_presence_frames:
                    LOG.info(
                        "Suppressed bird event: only %d crop(s) passed BioCLIP presence gate",
                        len(encoded),
                    )
                    continue

                encoded.sort(key=lambda item: legacy.image_sharpness(item.detection.crop), reverse=True)
                selected = encoded[: settings.sharpest_frames]
                preferred_names = legacy.preferred_regional_species_names(settings.region_profile, now.month)
                preferred = legacy.preferred_regional_species(settings.region_profile, now.month)
                broad = legacy.regional_species(settings.region_profile, now.month)
                plausible = legacy.identification_plausible_species(settings.region_profile, now.month)
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
                identification = legacy.resolve_identification(
                    frame_predictions,
                    settings.consensus_min_votes,
                    settings.species_min_confidence,
                    settings.species_min_margin,
                    plausible,
                )

                species = identification.candidate_name
                if not cooldown.is_ready(species, now):
                    LOG.info("Species cooldown active for %s", species)
                    continue
                image_path = legacy.save_bird_image(
                    selected[0].detection.crop, settings.image_dir, species, now
                )
                cooldown.mark_sent(species, now)
                try:
                    legacy.send_email(settings.email, identification, now, image_path)
                except Exception:
                    LOG.exception("Email failed; image remains saved")
                else:
                    LOG.info("Email alert sent for %s", identification.display_name)
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
        settings = legacy.load_settings()
        run(settings)
    except Exception:
        LOG.exception("Bird watcher could not start")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
