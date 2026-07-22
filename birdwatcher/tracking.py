"""Bird detection association and active-event deduplication."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


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
        math.hypot(
            max(ax2 - ax1, bx2 - bx1),
            max(ay2 - ay1, by2 - by1),
        ),
    )
    return distance / scale


def predict_box(previous_box: Box | None, current_box: Box) -> Box:
    """Project one step using the last observed center displacement."""
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
    """Score a likely continuation using overlap, motion prediction, and center distance."""
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
    relative_padding = min(
        1.0,
        crop_padding + 0.6 * (1.0 - min(1.0, box_dim / 220.0)),
    )
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
    return max(candidates, default=(0.0, 0.0, None), key=lambda item: (item[0], item[1]))[2]


def collect_tracked_crops(
    capture,
    models,
    initial_detection: TrackedDetection,
    settings,
    *,
    detect_fn: Callable = detect_birds,
    sleep_fn: Callable[[float], None] = time.sleep,
    clock_fn: Callable[[], float] = time.monotonic,
) -> list[TrackedDetection]:
    """Collect temporally spaced detections while following one spatial track."""
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
        matched = match_tracked_detection(
            current_box,
            detect_fn(models, frame, settings),
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
        return max(matches, default=(0.0, None), key=lambda item: item[0])[1]

    def partition_new_detections(
        self, detections: list[TrackedDetection], now: float
    ) -> list[TrackedDetection]:
        """Touch matching active events and return detections that represent new events."""
        self._prune(now)
        new_detections: list[TrackedDetection] = []
        for detection in sorted(detections, key=detection_quality, reverse=True):
            event = self._find_event(detection.box)
            if event is None:
                new_detections.append(detection)
                continue
            event.previous_box, event.box = event.box, detection.box
            event.last_seen_at = now
        return new_detections

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
