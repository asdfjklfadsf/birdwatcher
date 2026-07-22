"""Validation gates for detected bird events."""
from __future__ import annotations

import math
from statistics import median
from typing import Any, Callable


def validate_bird_event(
    detections: list[tuple[Any, float]],
    min_frames: int,
    min_median_confidence: float,
    max_aspect_ratio: float,
) -> tuple[list[tuple[Any, float]], str | None]:
    if min_frames < 1:
        raise ValueError("Minimum bird-event frame count must be positive")
    if not 0 < min_median_confidence <= 1:
        raise ValueError("Minimum event detector confidence must be in (0, 1]")
    if not math.isfinite(max_aspect_ratio) or max_aspect_ratio < 1:
        raise ValueError("Maximum bird-crop aspect ratio must be finite and at least 1")

    plausible = []
    rejected_shapes = 0
    for crop, score in detections:
        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            rejected_shapes += 1
            continue
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            rejected_shapes += 1
            continue
        plausible.append((crop, score))

    if len(plausible) < min_frames:
        reason = f"only {len(plausible)} valid detection(s), require {min_frames}"
        if rejected_shapes:
            reason += f"; rejected {rejected_shapes} crop(s) with implausible shape"
        return [], reason

    event_confidence = median(score for _, score in plausible)
    if event_confidence < min_median_confidence:
        return [], (
            f"median detector confidence {event_confidence:.1%} is below "
            f"{min_median_confidence:.1%}"
        )
    return plausible, None


def validate_visual_bird_presence(
    detections: list[tuple[Any, float]],
    scorer: Callable[[Any], float],
    min_score: float,
    min_frames: int,
) -> tuple[list[tuple[Any, float]], str | None, list[float]]:
    if not 0 <= min_score <= 1:
        raise ValueError("Minimum bird-presence score must be between 0 and 1")
    if min_frames < 1:
        raise ValueError("Minimum bird-presence frame count must be positive")

    accepted = []
    scores = []
    for crop, detector_score in detections:
        score = float(scorer(crop))
        scores.append(score)
        if score >= min_score:
            accepted.append((crop, detector_score))

    if len(accepted) < min_frames:
        median_score = median(scores) if scores else 0.0
        return [], (
            f"only {len(accepted)} crop(s) passed bird-presence score {min_score:.1%}, "
            f"require {min_frames}; median presence score {median_score:.1%}"
        ), scores
    return accepted, None, scores
