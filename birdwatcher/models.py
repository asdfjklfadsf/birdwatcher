"""Pinned detector and classifier model loading."""
from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

import cv2

LOG = logging.getLogger("bird_watcher")


def format_species_name(label: str) -> str:
    return label.strip().title()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_detector_model(model_name: str, expected_sha256: str) -> Path:
    path = Path(model_name).expanduser()
    if not path.exists():
        if model_name != "yolo11n.pt":
            raise FileNotFoundError(f"Detector model does not exist: {path}")
        url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
        temporary = path.with_suffix(path.suffix + ".download")
        LOG.info("Downloading pinned detector model from %s", url)
        try:
            urllib.request.urlretrieve(url, temporary)
            if file_sha256(temporary) != expected_sha256.lower():
                raise ValueError("Downloaded detector model checksum does not match DETECTOR_MODEL_SHA256")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if file_sha256(path) != expected_sha256.lower():
        raise ValueError(f"Detector model checksum mismatch: {path}")
    return path


class BirdModels:
    def __init__(
        self,
        detector_name: str,
        detector_sha256: str,
        classifier_name: str,
        classifier_revision: str,
        local_classifier_name: str = "imageomics/bioclip",
        local_classifier_revision: str = "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c",
    ):
        import open_clip
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from ultralytics import YOLO

        detector_path = prepare_detector_model(detector_name, detector_sha256)
        LOG.info("Loading verified bird detector: %s", detector_path)
        self.detector = YOLO(str(detector_path))
        LOG.info("Loading species classifier: %s", classifier_name)
        self.processor = AutoImageProcessor.from_pretrained(
            classifier_name, revision=classifier_revision, use_fast=False
        )
        self.classifier = AutoModelForImageClassification.from_pretrained(
            classifier_name, revision=classifier_revision, use_safetensors=True
        )
        self.classifier.eval()
        self.torch = torch
        self.device = torch.device("cpu")
        self.classifier.to(self.device)

        LOG.info("Loading local BioCLIP classifier: %s", local_classifier_name)
        local_snapshot = snapshot_download(
            repo_id=local_classifier_name,
            revision=local_classifier_revision,
            allow_patterns=[
                "open_clip_config.json",
                "open_clip_pytorch_model.bin",
                "merges.txt",
                "vocab.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
        local_source = f"local-dir:{local_snapshot}"
        self.local_classifier, _, self.local_preprocess = open_clip.create_model_and_transforms(
            local_source,
            require_pretrained=True,
            weights_only=True,
        )
        self.local_tokenizer = open_clip.get_tokenizer(local_source)
        self.local_classifier.eval().to(self.device)
        LOG.info("Models ready; species classifiers device: %s", self.device)

    def collect_bird_boxes(self, frame, imgsz: int, confidence: float):
        """Return scored COCO-bird boxes for one frame or tile."""
        results = self.detector.predict(
            source=frame,
            classes=[14],
            conf=confidence,
            imgsz=imgsz,
            verbose=False,
        )
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            score = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, score))
        return boxes

    # Retained for pre-refactor callers that used the private spelling.
    _collect_bird_boxes = collect_bird_boxes

    def identify_species_candidates(self, bird_image, top_k: int = 3) -> list[tuple[str, float]]:
        from PIL import Image

        rgb_image = cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb_image), return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            logits = self.classifier(**inputs).logits
            probabilities = self.torch.softmax(logits, dim=-1)[0]
            count = min(top_k, int(probabilities.shape[0]))
            confidences, class_ids = self.torch.topk(probabilities, count)
        return [
            (
                format_species_name(self.classifier.config.id2label[int(class_id)]),
                float(confidence),
            )
            for confidence, class_id in zip(confidences.tolist(), class_ids.tolist())
        ]
