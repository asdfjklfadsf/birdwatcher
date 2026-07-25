"""Camera and image persistence helpers."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import cv2


def image_sharpness(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "bird"


def save_bird_image(image, directory: Path, species: str, observed_at: datetime) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    resize = getattr(cv2, "resize", None)
    if resize is not None and hasattr(image, "shape"):
        height, width = image.shape[:2]
        short_side = min(height, width)
        target_short_side = 320
        max_long_side = 1280
        if short_side < target_short_side:
            scale = target_short_side / max(1, short_side)
            if max(height, width) * scale > max_long_side:
                scale = max_long_side / max(height, width)
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            image = resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    filename = f"{observed_at:%Y%m%d_%H%M%S_%f}_{safe_filename(species)}.jpg"
    path = directory / filename
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not save image to {path}")
    return path


def open_camera(camera: int | str, width: int, height: int, fps: float):
    if isinstance(camera, str) and camera.startswith("/dev/"):
        capture = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    # Scanning is slower than the capture rate, so a queued buffer would hand us
    # frames from seconds ago. Ask for the newest frame instead; not every
    # backend honors this, so it is best-effort.
    buffer_size = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffer_size is not None:
        capture.set(buffer_size, 1)
    if not capture.isOpened():
        capture.release()
        return None
    return capture
