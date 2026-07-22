"""Real-dependency startup smoke test without downloading model weights."""
from __future__ import annotations

import importlib
import os
from pathlib import Path

import numpy as np


def main() -> None:
    required_modules = (
        "cv2",
        "torch",
        "transformers",
        "ultralytics",
        "open_clip",
        "huggingface_hub",
        "PIL",
    )
    loaded = {name: importlib.import_module(name) for name in required_modules}

    cv2 = loaded["cv2"]
    torch = loaded["torch"]
    transformers = loaded["transformers"]
    ultralytics = loaded["ultralytics"]
    open_clip = loaded["open_clip"]

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    assert converted.shape == frame.shape
    assert float(torch.tensor([1.0, 2.0]).sum()) == 3.0
    assert hasattr(transformers, "AutoImageProcessor")
    assert hasattr(transformers, "AutoModelForImageClassification")
    assert callable(ultralytics.YOLO)
    assert callable(open_clip.create_model_and_transforms)
    assert callable(open_clip.get_tokenizer)

    # Production package files must not depend on the compatibility facade.
    offenders = []
    for path in Path("birdwatcher").glob("*.py"):
        if "legacy_main" in path.read_text(encoding="utf-8"):
            offenders.append(path.as_posix())
    assert not offenders, f"production modules still reference legacy_main: {offenders}"

    os.environ.setdefault("SMTP_HOST", "smtp.example.invalid")
    os.environ.setdefault("EMAIL_FROM", "birdwatcher@example.invalid")
    os.environ.setdefault("EMAIL_TO", "recipient@example.invalid")

    from birdwatcher import alerts, app, classification, config, media, models, region, tracking

    runtime = config.load_runtime_config()
    assert runtime.event_clear_seconds == 6.0
    assert runtime.settings.detector_model == "yolo11n.pt"
    assert callable(models.BirdModels)
    assert callable(media.open_camera)
    assert callable(alerts.EmailRetryQueue)
    assert callable(classification.hybrid_predictions)
    assert callable(region.resolve_identification)
    assert callable(tracking.ActiveEventTracker)
    assert callable(app.main)

    print("Runtime dependency smoke test passed")


if __name__ == "__main__":
    main()
