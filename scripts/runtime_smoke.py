"""Real-dependency startup smoke test without downloading model weights."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


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

    from birdwatcher import (
        alerts,
        app,
        classification,
        config,
        media,
        models,
        region,
        species,
        tracking,
    )

    runtime = config.load_runtime_config()
    assert runtime.event_clear_seconds == 6.0
    assert runtime.settings.detector_model == "yolo11n.pt"
    assert runtime.settings.tile_sweep_interval == 5.0

    # Every species comparison must agree on one identity per bird.
    assert species.canonical_species_key("Tit Mouse") == species.canonical_species_key(
        "Tufted Titmouse"
    )
    plausible = region.identification_plausible_species(runtime.settings.region_profile, 7)
    assert species.canonical_species_key("Tit Mouse") in plausible
    assert classification.combine_classifier_predictions is region.combine_classifier_predictions
    assert hasattr(models.BirdModels, "collect_bird_boxes")
    assert callable(tracking.deduplicate_boxes)
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
