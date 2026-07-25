"""Bird detection association and active-event deduplication."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .constants import DUPLICATE_DETECTION_IOU

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackedDetection:
    crop: Any
    score: float
    box: Box


@dataclass
class ActiveEvent:
    previous_box: Box | None
    box: Box
    started_at: float
    last_seen_at: float


def box_iou(a: Box, b: Box) -> float:
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


def center_distance_ratio(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    distance = math.hypot(acx - bcx, acy - bcy)
    scale = max(
        1.0,
        math.hypot(max(ax2 - ax1, bx2 - bx1), max(ay2 - ay1, by2 - by1)),
    )
    return distance / scale


def predict_box(previous_box: Box | None, current_box: Box) -> Box:
    if previous_box is None:
        return current_box
    px1, py1, px2, py2 = previous_box
    cx1, cy1, cx2, cy2 = current_box
    previous_cx, previous_cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    current_cx, current_cy = (cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0
    dx, dy = current_cx - previous_cx, current_cy - previous_cy
    return (
        int(round(cx1 + dx)),
        int(round(cy1 + dy)),
        int(round(cx2 + dx)),
        int(round(cy2 + dy)),
    )


def box_match_score(current_box: Box, candidate_box: Box, previous_box: Box | None = None) -> float | None:
    predicted = predict_box(previous_box, current_box)
    current_iou = box_iou(current_box, candidate_box)
    predicted_iou = box_iou(predicted, candidate_box)
    predicted_distance = center_distance_ratio(predicted, candidate_box)
    current_distance = center_distance_ratio(current_box, candidate_box)
    if (
        current_iou < 0.05
        and predicted_iou < 0.05
        and predicted_distance > 1.0
        and current_distance > 1.25
    ):
        return None
    return (
        1.5 * predicted_iou
        + 0.5 * current_iou
        - 0.30 * predicted_distance
        - 0.10 * current_distance
    )


def pad_crop(frame, box: Box, crop_padding: float):
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


def deduplicate_boxes(
    boxes: list[tuple[int, int, int, int, float]],
    iou_threshold: float = DUPLICATE_DETECTION_IOU,
) -> list[tuple[int, int, int, int, float]]:
    """Greedy non-maximum suppression over scored boxes, highest score first.

    The full-frame and tiled sweeps overlap, so one bird can be reported several
    times. Without this, each copy becomes its own detection and event.
    """
    kept: list[tuple[int, int, int, int, float]] = []
    for candidate in sorted(boxes, key=lambda item: -item[4]):
        box = candidate[:4]
        if any(box_iou(box, keep[:4]) > iou_threshold for keep in kept):
            continue
        kept.append(candidate)
    return kept


def detect_birds(
    models,
    frame,
    settings,
    *,
    allow_tile_sweep: bool = True,
    tile_sweep: TileSweepThrottle | None = None,
    now: float | None = None,
) -> list[TrackedDetection]:
    """Detect birds, falling back to a low-confidence sweep when none are found.

    Both knobs gate only the nine-tile pass, which costs nine extra inferences;
    the cheaper full-frame passes always run. ``allow_tile_sweep`` is a hard off
    switch, while ``tile_sweep`` is a rate limiter shared across the pipeline.

    The limiter is consulted here rather than by the caller because only this
    function knows whether a sweep is actually about to happen. Deciding up
    front would spend the budget on frames where the cheaper passes already
    found the bird, leaving none for the frames that lost it.
    """
    height, width = frame.shape[:2]
    raw: list[tuple[int, int, int, int, float]] = []

    def accept(x1: int, y1: int, x2: int, y2: int, score: float) -> None:
        box_width, box_height = max(0, x2 - x1), max(0, y2 - y1)
        if not box_width or not box_height:
            return
        ratio = max(box_width / box_height, box_height / box_width)
        if ratio <= settings.max_bird_crop_aspect_ratio:
            raw.append((x1, y1, x2, y2, score))

    for x1, y1, x2, y2, score in models.collect_bird_boxes(
        frame, 640, settings.detection_confidence
    ):
        accept(x1, y1, x2, y2, score)

    if not raw:
        for x1, y1, x2, y2, score in models.collect_bird_boxes(
            frame, 1280, settings.detection_floor_confidence
        ):
            accept(x1, y1, x2, y2, score)

    if (
        not raw
        and allow_tile_sweep
        and (tile_sweep is None or tile_sweep.allow(time.monotonic() if now is None else now))
    ):
        for ty in range(3):
            for tx in range(3):
                y0, y1t = int(ty * height / 3), int((ty + 1) * height / 3)
                x0, x1t = int(tx * width / 3), int((tx + 1) * width / 3)
                tile = frame[y0:y1t, x0:x1t]
                if tile.size == 0:
                    continue
                for lx1, ly1, lx2, ly2, score in models.collect_bird_boxes(
                    tile, 640, settings.detection_floor_confidence
                ):
                    accept(x0 + lx1, y0 + ly1, x0 + lx2, y0 + ly2, score)
        # Only the tiled pass merges results from more than one inference, so it
        # is the only place a bird could be reported twice. Every other pass
        # returns boxes the detector already ran NMS over, and suppressing those
        # again would discard second birds the detector deliberately kept.
        raw = deduplicate_boxes(raw)

    return [
        TrackedDetection(
            crop=pad_crop(frame, (x1, y1, x2, y2), settings.crop_padding),
            score=score,
            box=(x1, y1, x2, y2),
        )
        for x1, y1, x2, y2, score in raw
    ]


def detection_quality(detection: TrackedDetection) -> float:
    x1, y1, x2, y2 = detection.box
    return max(0, x2 - x1) * max(0, y2 - y1) * detection.score


def choose_initial_detection(detections: Iterable[TrackedDetection]) -> TrackedDetection | None:
    detections = list(detections)
    return max(detections, key=detection_quality) if detections else None


def match_tracked_detection(
    current_box: Box,
    detections: list[TrackedDetection],
    previous_box: Box | None = None,
) -> TrackedDetection | None:
    candidates: list[tuple[float, float, TrackedDetection]] = []
    for detection in detections:
        score = box_match_score(current_box, detection.box, previous_box)
        if score is not None:
            candidates.append((score, detection.score, detection))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


class TileSweepThrottle:
    """Rate-limit the nine-tile sweep across every stage of the pipeline.

    Top-level scans and tracked bursts share one instance, so a burst cannot
    spend nine extra inferences per frame while the idle loop is being polite.
    Time is supplied by the caller, which already reads a clock.
    """

    def __init__(self, interval_seconds: float):
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("Tile sweep interval must be finite and nonnegative")
        self.interval_seconds = interval_seconds
        self._next_allowed = 0.0

    def allow(self, now: float) -> bool:
        if now < self._next_allowed:
            return False
        self._next_allowed = now + self.interval_seconds
        return True


def collect_tracked_crops(
    capture,
    models,
    initial_detection: TrackedDetection,
    settings,
    *,
    detect_fn: Callable = detect_birds,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
    tile_sweep: TileSweepThrottle | None = None,
) -> list[TrackedDetection]:
    detections = [initial_detection]
    previous_box: Box | None = None
    current_box = initial_detection.box
    next_sample = clock_fn() + settings.burst_frame_interval
    for _ in range(settings.burst_frames - 1):
        delay = next_sample - clock_fn()
        if delay > 0:
            sleep_fn(delay)
        next_sample += settings.burst_frame_interval
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        # A nine-frame burst would otherwise be able to run the tile sweep eight
        # times, stretching the sampling schedule well past its wall-clock plan.
        matched = match_tracked_detection(
            current_box,
            detect_fn(models, frame, settings, tile_sweep=tile_sweep, now=clock_fn()),
            previous_box,
        )
        if matched is None:
            continue
        detections.append(matched)
        previous_box, current_box = current_box, matched.box
    return detections


class ActiveEventTracker:
    """Suppress repeated work for birds that are still part of an active scene event."""

    def __init__(self, clear_seconds: float, max_age_seconds: float):
        if clear_seconds <= 0 or max_age_seconds <= 0:
            raise ValueError("Active event timeouts must be positive")
        self.clear_seconds = clear_seconds
        self.max_age_seconds = max_age_seconds
        self._events: list[ActiveEvent] = []

    @property
    def active_count(self) -> int:
        return len(self._events)

    def _prune(self, now: float) -> None:
        self._events = [
            event
            for event in self._events
            if now - event.last_seen_at <= self.clear_seconds
            and now - event.started_at <= self.max_age_seconds
        ]

    def _find_event(self, box: Box) -> ActiveEvent | None:
        matches: list[tuple[float, ActiveEvent]] = []
        for event in self._events:
            score = box_match_score(event.box, box, event.previous_box)
            if score is not None:
                matches.append((score, event))
        if not matches:
            return None
        return max(matches, key=lambda item: item[0])[1]

    def partition_new_detections(
        self, detections: list[TrackedDetection], now: float
    ) -> list[TrackedDetection]:
        """Match active events one-to-one and return detections that are new events."""
        self._prune(now)
        ranked = sorted(detections, key=detection_quality, reverse=True)
        candidates: list[tuple[float, float, float, int, int]] = []
        for event_index, event in enumerate(self._events):
            for detection_index, detection in enumerate(ranked):
                score = box_match_score(event.box, detection.box, event.previous_box)
                if score is not None:
                    candidates.append(
                        (score, detection.score, detection_quality(detection), event_index, detection_index)
                    )

        used_events: set[int] = set()
        used_detections: set[int] = set()
        for _score, _confidence, _quality, event_index, detection_index in sorted(
            candidates, reverse=True
        ):
            if event_index in used_events or detection_index in used_detections:
                continue
            event = self._events[event_index]
            detection = ranked[detection_index]
            event.previous_box, event.box = event.box, detection.box
            event.last_seen_at = now
            used_events.add(event_index)
            used_detections.add(detection_index)

        return [detection for index, detection in enumerate(ranked) if index not in used_detections]

    def mark_event(self, box: Box, now: float, previous_box: Box | None = None) -> None:
        self._prune(now)
        event = self._find_event(box)
        if event is not None:
            event.previous_box, event.box = event.box, box
            event.last_seen_at = now
            return
        self._events.append(
            ActiveEvent(
                previous_box=previous_box,
                box=box,
                started_at=now,
                last_seen_at=now,
            )
        )

    def observe_no_detection(self, now: float) -> None:
        self._prune(now)
